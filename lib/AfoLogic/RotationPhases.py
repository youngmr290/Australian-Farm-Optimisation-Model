"""

This module reads in rotation information that has been generated from RotGeneration.py and manipulates
it to produce the rotation parameters.

The flexible and detailed handling of rotation decisions allows the representation of paddock history on the
current landuse to be represented as well as the tactical and strategical decisions that accompany rotation. Such
as delaying the choice of rotation phase at the start of the growing season and changing rotation phase later in the
growing season. Or even adding a summer crop if the season permits.

author: young
"""

#python modules
import pandas as pd
import numpy as np
na = np.newaxis
import timeit

#AFO modules - a bunch of other precalc modules import this module
from . import StructuralInputs as sinp
from . import PropertyInputs as pinp
from . import Functions as fun
from . import Periods as per
from . import SeasonalFunctions as zfun


def f_v_phase_increment_adj(param, p7_pos, z_pos, p5_pos=None, numpy=False):
    '''
    Adjust v_phase param for v_phase_change_increase.

    v_phase_change_increase must incur the requirement to date for labour and cash for the phase.
    This is making the assumption that any jobs carried out, and any expenditure
    (fertiliser or chemical applied) will be applied even though the phase is selected later in the year.
    This stops the optimisation selecting the phase in the last node and receiving the income without
    incurring any costs.

    Note 1: Yield and stubble do not require increment params because it is not possible to harvest a
    rotation before the rotation is selected.

    Note 2: labour gets handled slightly different. Labour that occurs in previous labour periods before the season
        period when v_phase_change_increase is selected must be completed in the first labour period when the phase is selected.

    Note 3: f_v_phase_increment_adj starts counting from the start of season but there are no phase costs/labour between until
        after the break for each season otherwise costs would get double counted in
        medium/late breaks where phases are carried over past the start of the season to provide dry pas and stubble area.


    :param param: numpy array or pandas series - parameter with p7 axis.
    :param p7_pos: negative int: axis/level of p7
    :param z_pos: negative int: axis/level of z
    :param p5_pos: optional negative int: axis/level of p5 axis.
    :param numpy: Boolean, stating if param is numpy.
    '''
    ##get p7 periods into corect shape
    p7_date_p7z = per.f_season_periods()[:-1,...] #slice off end date p7
    if p7_pos > z_pos:
        p7_date_p7z = np.swapaxes(p7_date_p7z,0,1) #handle if z axis is before p7 axis
        p7_date_p7zetc = fun.f_expand(p7_date_p7z, left_pos=p7_pos, right_pos2=p7_pos, left_pos2=z_pos)
    else:
        p7_date_p7zetc = fun.f_expand(p7_date_p7z, left_pos=z_pos, right_pos2=z_pos, left_pos2=p7_pos)

    ##convert pd.Series to numpy
    if not numpy:
        ##store index
        index = param.index
        ##reshape array to be numpy
        reshape_size = param.index.remove_unused_levels().levshape  # create a tuple with the rights dimensions
        param = np.reshape(param.values,reshape_size)

    ##uncluster z so that cumsum works correctly (if a z is clustered labour/cost is still needed in that z for the cumsum)
    maskz8_p7z = zfun.f_season_transfer_mask(p7_date_p7zetc,z_pos=z_pos,mask=True)
    index_zetc = fun.f_expand(np.arange(maskz8_p7z.shape[z_pos]), z_pos)
    a_zcluster_p7zetc = np.maximum.accumulate(index_zetc * maskz8_p7z, axis=z_pos)
    a_zcluster = np.broadcast_to(a_zcluster_p7zetc, param.shape)
    param = np.take_along_axis(param, a_zcluster, axis=z_pos)

    ##calc cost to date - occurs 0 in the current period because v_phase incurs current period cost.
    param_increment = np.roll(np.cumsum(param, axis=p7_pos),1, axis=p7_pos)
    slc = [slice(None)] * len(param_increment.shape)
    slc[p7_pos] = slice(0,1)
    param_increment[tuple(slc)] = 0

    ##handle labour period axis if it exists
    if p5_pos:
        ##get p5 periods into correct shape
        p5_date_p5z = per.f_p_dates_df().values[:-1,...]  # slice off end date p5
        if p5_pos > z_pos:
            p5_date_p5z = np.swapaxes(p5_date_p5z,0,1)  # handle if z axis is before p7 axis
            p5_date_p5zetc = fun.f_expand(p5_date_p5z,left_pos=p5_pos,right_pos2=p5_pos,left_pos2=z_pos)
        else:
            p5_date_p5zetc = fun.f_expand(p5_date_p5z,left_pos=z_pos,right_pos2=z_pos,left_pos2=p5_pos)

        ###labour period that is start of p7 node
        p5_is_start_p7_p5p7zetc = p7_date_p7zetc == p5_date_p5zetc

        ###create temp variable which has the total labour for a given p7 for each p5
        temp_param_increment = np.cumsum(param_increment,axis=p5_pos)
        ###mask only for p5 which are start of p7. This means that any labour prior to the start of the node
        ### must be completed in the first node when the phase is selected.
        param_increment = temp_param_increment * p5_is_start_p7_p5p7zetc

    ##add index if pandas
    if not numpy:
        param_increment = pd.Series(param_increment.ravel(), index=index)

    return param_increment


def f_season_params(params):
    '''
    Create params for phase period transfer.
    '''
    ##inputs
    keys_p7 = per.f_season_periods(keys=True)
    keys_z = zfun.f_keys_z()
    phases_df = pinp.phases_r
    landuse_r = phases_df.iloc[:,-1].values
    dry_sown_landuses = sinp.landuse['dry_sown']
    phase_is_drysown_r = np.any(landuse_r[:,na]==list(dry_sown_landuses), axis=-1)

    ##z8z9 transfer
    start_phase_periods_p7z = per.f_season_periods()[:-1,:] #remove end date of last period
    season_start_z = per.f_season_periods()[0,:] #slice season node to get season start
    period_is_seasonstart_p7z = start_phase_periods_p7z==season_start_z
    mask_provwithinz8z9_p7z8z9, mask_provbetweenz8z9_p7z8z9, mask_reqwithinz8_p7z8, mask_reqbetweenz8_p7z8 = zfun.f_season_transfer_mask(
        start_phase_periods_p7z, period_is_seasonstart_pz=period_is_seasonstart_p7z, z_pos=-1) #the req masks don't do the correct job for rotation and hence are not used.

    ##for the rot_hisory_within constraint the parentz prov param needs to be different.
    ## within season ancestor provide. This is different to parent provide because in the dual growing season it is likely
    ## that each growing season will contain more than one p7 slice. In the within year history constraint
    ## gs0 provides gs2 therefore if gs0 is two p7 long then it is like grandparent provide. Therefore we have to create
    ## a new z8z9 provide param
    index_z = np.arange(len(keys_z))
    mask_childz8_p7z8 = zfun.f_season_transfer_mask(start_phase_periods_p7z, z_pos=-1, mask=True)
    mask_ancestor_provwithinz8z9_p7z8z9 = np.maximum.accumulate(mask_childz8_p7z8 * index_z, axis=1)[:,na,:] == index_z[:,na]

    # ###for rotation the between and within constraints are acting on different things (history vs the acutal phase) therefore
    # ### the req params above don't work because they have been adjusted for season start. so in the following line i make a
    # ### new req param which doesn't account for within or between
    # mask_childz8_p7z8 = zfun.f_season_transfer_mask(start_phase_periods_p7z,z_pos=-1,mask=True)

    # ##mask phases which transfer in each m
    # if pinp.general['steady_state'] or np.count_nonzero(pinp.general['i_mask_z']) == 1:
    #     ###if steady state then there is no m transfering
    #     mask_phases_rm = np.zeros((len(phases_df),len(keys_p7)))
    # else:
    #     ###if dsp no transfer at the end of yr to the start (different for dry sown landuses since m-1 is essentially the start for them)
    #     mask_phases_rm = np.ones((len(phases_df),len(keys_p7)))
    #     mask_phases_rm[:,-1] = phase_is_drysown_r #only dry sown landuse pass from m[-1] to m[0] because m[-1] is the period when dry sown phases are selected.
    #     mask_phases_rm[:,-2] = np.logical_not(phase_is_drysown_r) #v_phase dry does not provide into m[-1]. if the model wants dry sown phases it can select via v_phase_change_increase.
    #
    # ##dry seeding link between season - dry seeding must happen in all seasons that brk after the season with dry seeding.
    # ##the same amount of dry seeding must occur for all seasons with the same break therefore the end season passes back to the start for a given brk.
    # ##the param below is used on the require side of phase_increment. It says if you want to dry sow in z0 you must
    # ## dry sow in z1 and if you want to sow in z1 you must dry sow in z2 etc.
    # ## The second part of the mask creation makes it so that the same dry sowing occurs for each season that has the same brk.
    # mask_drynext_z8z9 = index_z[:,na] == index_z-1 #every season must have at least the same amount of dry sowing as the previous season.
    # mask_drystart_z8z9 = (initiating_parent_z != np.roll(initiating_parent_z,-1))[:,na]*(index_z==initiating_parent_z[:,na])#each season with the same brk must have the same amount of dry sowiing.
    # ###only for dry sown phases
    # mask_drynext_z8z9 = mask_drynext_z8z9 * phase_is_drysown_r[:,na,na]
    # mask_drystart_z8z9 = mask_drystart_z8z9 * phase_is_drysown_r[:,na,na]

    ##build params
    arrays_p7z8z9 = [keys_p7, keys_z, keys_z]

    arrays_p7z8 = [keys_p7, keys_z]

    params['p_mask_childz_within_phase'] = fun.f1_make_pyomo_dict(mask_reqwithinz8_p7z8*1, arrays_p7z8)
    params['p_mask_childz_between_phase'] = fun.f1_make_pyomo_dict(mask_reqbetweenz8_p7z8*1, arrays_p7z8)
    params['p_parentz_provwithin_phase'] = fun.f1_make_pyomo_dict(mask_provwithinz8z9_p7z8z9*1, arrays_p7z8z9)
    params['p_parentz_provbetween_phase'] = fun.f1_make_pyomo_dict(mask_provbetweenz8z9_p7z8z9*1, arrays_p7z8z9)
    params['p_ancestorz_provwithinz_phase'] = fun.f1_make_pyomo_dict(mask_ancestor_provwithinz8z9_p7z8z9*1, arrays_p7z8z9)
    # params['p_mask_phases'] =dict(zip(tup_rm, mask_phases_rm.ravel()*1))
    # params['p_dryz_link'] =dict(zip(tup_rz8z9, mask_drynext_z8z9.ravel()*1))
    # params['p_dryz_link2'] =dict(zip(tup_rz8z9, mask_drystart_z8z9.ravel()*1))


def f_landuses_phases(params,r_vals):
    '''
    * Read in the rotation list generated by RotGeneration.py
    * Create rotation area parameter for pyomo (simply each rotation phase uses 1ha of area).
    * Store rotation list and pasture phases list to report dictionary

    '''
    phases=pinp.phases_r
    phases_rk = phases.set_index(phases.columns[-1], append=True) #add landuse as index level
    params['phases_rk'] = dict.fromkeys(phases_rk.index,1)

    ##store r_vals
    fun.f1_make_r_val(r_vals,phases,'phases')
    fun.f1_make_r_val(r_vals,sinp.landuse['All_pas'],'all_pastures')#all_pas2 includes the cont pasture landuses
    fun.f1_make_r_val(r_vals,sinp.landuse['E'],'all_cereals')#all_pas2 includes the cont pasture landuses
    fun.f1_make_r_val(r_vals,sinp.landuse['N'],'all_canolas')#all_pas2 includes the cont pasture landuses


def f_rot_lmu_params(params):
    '''
    Create parameters for the total area available on each LMU and the total area that is never cropped on each lmu due
    to topography, paddock location or farmer preference.

    '''
    ##area
    lmu_mask = pinp.general['i_lmu_area'] > 0
    params['lmu_area'] = dict(zip(pinp.general['i_lmu_idx'][lmu_mask], pinp.general['i_lmu_area'][lmu_mask]))
    params['p_not_cropable_area_l'] = dict(zip(pinp.general['i_lmu_idx'][lmu_mask], pinp.general['i_non_cropable_area_l'][lmu_mask]))


def f_phase_link_params(params):
    '''
    Create parameters for phase link constraint. These parameters mask when phase can be changed and also force
    a change of phase at the break of each season (change is required because some costs e.g. seeding are connected
    to v_phase_change_increase.

    Phases from p7_prev dont transfer in the p7 period immediately preceding the break of season
    for each weather-year (z). To force a v_phase_change (to current season land-use or to PNC) at the break.
    Dry sown phases can't transfer between seasons but they can at break of the medium and late seasons.

    Note: a2 phases dont provide a history. This is required to stop pnc being selected all growing season
    (if the model wants pasture it has to change to normal pasture and will incur any prior costs).
    '''
    ##inputs
    dry_sown_landuses = sinp.landuse['dry_sown']
    phases_df = pinp.phases_r
    landuse_r = phases_df.iloc[:,-1].values
    keys_k  = np.asarray(list(sinp.landuse['All']))  #landuse
    phases_rotn_df = pinp.phases_r
    keys_p7 = per.f_season_periods(keys=True)
    keys_r = np.array(phases_rotn_df.index).astype('str')
    keys_z = zfun.f_keys_z()
    i_break_z = zfun.f_seasonal_inp(pinp.general['i_break'], numpy=True)

    ##if pasture sowing can occur beore season break then need to add sown pasture landuses to dry sown list
    for pasture in sinp.general['pastures'][pinp.general['pas_inc_t']]:
        resown_pas = sinp.landuse['resown_pasture_sets'][pasture]
        start_pas_seeding = zfun.f_seasonal_inp(pinp.pasture_inputs[pasture]['Date_Seeding'],numpy=True)
        if any(start_pas_seeding<i_break_z):
            dry_sown_landuses = dry_sown_landuses | resown_pas


    ##p_phase_area_transfers is a True/False and is False in the p7 period immediately preceding the break of season
    ## for each weather-year (z). To force a v_phase_change (to current season land-use or to PNC) at the break.
    ## Dry sown phases can't transfer between seasons but they can at break of the medium and late seasons.
    ###first calculate which p7 is break for each season
    start_date_p7z = per.f_season_periods()[:-1, :]  # remove end date of last period
    end_date_p7z = per.f_season_periods()[1:, :]
    next_period_is_break_p7z = np.roll(np.logical_and(start_date_p7z<=i_break_z, i_break_z<end_date_p7z), shift=-1, axis=0) #have to do it this way because for 'typ' break of season may not be a node.
    next_period_isnot_break_p7z = np.logical_not(next_period_is_break_p7z)
    ###first calculate which p7 is break for each season
    next_period_is_seasonstart_p7z = np.roll(start_date_p7z==start_date_p7z[0,:], shift=-1, axis=0) #have to do it this way because for 'typ' break of season may not be a node.
    next_period_isnot_seasonstart_p7z = np.logical_not(next_period_is_seasonstart_p7z)
    ###third calculate which phases are dry sown
    phase_is_drysown_r = np.any(landuse_r[:,na]==list(dry_sown_landuses), axis=-1)
    ###calculate if period is transfer at season break this is false for everything except dry sown phases
    transfer_break_p7zr = np.logical_or(next_period_isnot_break_p7z[...,na], phase_is_drysown_r)
    ###calculate if period is transfer at season start - this is true for all phases except dry sown ones
    transfer_seasonstart_p7zr =np.logical_or(next_period_isnot_seasonstart_p7z[...,na], np.logical_not(phase_is_drysown_r))
    ###combine
    p_phase_area_transfers_p7zr = np.logical_and(transfer_break_p7zr, transfer_seasonstart_p7zr)

    ##create mask to control what phases can be changed in each p7
    phase_can_increase_kp7 = pinp.general['i_phase_can_increase_kp7'] #input to control what landuses can change_increase in each p7
    phase_can_reduce_kp7 = pinp.general['i_phase_can_reduce_kp7'] #input to control what landuses can change_reduce in each p7
    ###only pnc can reduce at season brk nodes. This stops phases increasing and reducing to get poc and crop grazing.
    ### This line of code saves adding a z axis to the input
    p7_is_between_seasonbrk_and_endbrk_p7z = np.logical_or(start_date_p7z < i_break_z, np.max(i_break_z) < start_date_p7z)
    phase_can_reduce_kp7z = phase_can_reduce_kp7[...,na] * np.logical_or(p7_is_between_seasonbrk_and_endbrk_p7z, keys_k[:,na,na]=='a2')
    ###change k to r
    landuse_r = phases_rotn_df.iloc[:, -1].values
    a_k_rk = landuse_r[:, na] == keys_k
    phase_can_increase_p7r = np.sum(phase_can_increase_kp7 * a_k_rk[...,na], axis=1).T
    phase_can_reduce_rp7z = np.sum(phase_can_reduce_kp7z * a_k_rk[...,na,na], axis=1)
    ###stop phases increasing in the period from season start to break of season, except dry sown landuses. This is
    ### required because the history constraint doesnt exist between season start and break of season so that last yrs
    ### phase can cary over in the medium and later break so that dry pasture and stubble can still be grazed
    season_broken_p7z = end_date_p7z > i_break_z #has the season broken before the end of the given p7
    phase_can_increase_before_brk_p7zr = np.logical_or(phase_is_drysown_r, season_broken_p7z[...,na])
    phase_can_increase_p7zr = np.logical_and(phase_can_increase_p7r[:,na,:], phase_can_increase_before_brk_p7zr)

    ##make params
    arrays_rp7z = [ keys_r, keys_p7, keys_z]
    arrays_p7zr = [keys_p7, keys_z, keys_r]

    params['p_phase_area_transfers_p7zr'] = fun.f1_make_pyomo_dict(p_phase_area_transfers_p7zr*1, arrays_p7zr)
    params['p_phase_can_increase_p7zr'] = fun.f1_make_pyomo_dict(phase_can_increase_p7zr*1, arrays_p7zr)
    params['p_phase_can_reduce_rp7z'] = fun.f1_make_pyomo_dict(phase_can_reduce_rp7z*1, arrays_rp7z)


def f_rot_hist_params(params):
    '''
    Create parameters for landuse history provided and required by each rotation phase.

    '''
    ##inputs
    keys_p7 = per.f_season_periods(keys=True)
    keys_z = zfun.f_keys_z()
    index_p7 = np.arange(len(keys_p7))

    ##Mask to skip constraint in the period from season start to break of season. This is
    ## required so that last yrs phase can cary over in the medium and later break so that dry pasture and stubble can still be grazed
    i_break_z = zfun.f_seasonal_inp(pinp.general['i_break'], numpy=True)
    end_date_p7z = per.f_season_periods()[1:, :]
    season_broken_p7z = end_date_p7z > i_break_z #has the season broken before the end of the given p7

    ##mask to control which p7 are constrained by growing season 0
    p7_end_gs0 = index_p7[pinp.general['i_gs_p7_end'][0]]  # p7 period from growing season 0.
    p7_constrained_gs0_p7 = index_p7 > p7_end_gs0

    ##mask to control which p7 are constrained by growing season 1
    p7_constrained_gs1_p7 = index_p7 <= p7_end_gs0

    ##combine the growing season mask and the season broken mask
    p_inc_hist_gs0_con_p7z = np.logical_and(p7_constrained_gs0_p7[:,na], season_broken_p7z)
    p_inc_hist_gs1_con_p7z = np.logical_and(p7_constrained_gs1_p7[:,na], season_broken_p7z)


    ##history prov and req
    rot_req = pinp.rot_req.set_index([0,1])
    rot_prov = pinp.rot_prov.set_index([0,1])
    params['hist_prov'] = rot_prov.squeeze().to_dict()
    params['hist_req'] = rot_req.squeeze().to_dict()

    ##create constraint mask - this is required when some rotations have been masked out (e.g unprofitbale rotation) - when rot are masked out it can result in nothing requiring a history therefore meaning the constraint needs to be skipped
    phases_r = pinp.phases_r.index #list of phases after the rot mask has been applied
    masked_rot_req = rot_req[rot_req.index.get_level_values(0).isin(phases_r)] #mask out the removed rotations from req param
    req_hist = masked_rot_req.index.get_level_values(1).unique() #get the unique histories after rot mask
    ###histories that are required by any rotations
    mask_hist = pinp.s_rotcon1.index.isin(req_hist)
    params['hist_used'] = dict(zip(pinp.s_rotcon1.index, mask_hist))

    arrays_p7z = [keys_p7, keys_z]
    params['p_inc_hist_gs0_con_p7z'] = fun.f1_make_pyomo_dict(p_inc_hist_gs0_con_p7z*1, arrays_p7z)
    params['p_inc_hist_gs1_con_p7z'] = fun.f1_make_pyomo_dict(p_inc_hist_gs1_con_p7z*1, arrays_p7z)

def f_rot_hist4_params(params):
    '''
    History 4 constraint is used to ensure dual landuse follows the correct part a landuse.

    '''
    keys_k  = np.asarray(list(sinp.landuse['All']))  #landuse
    phases_rotn_df = pinp.phases_r

    ##phase is dual - used to skip constraint
    landuse_is_dual_k = pd.Series(data=sinp.general['i_landuse_is_dual'], index=keys_k, name='dual')
    params['phase_is_dual_r'] = landuse_is_dual_k.to_dict()

    ##hist4 req
    hist4_req_k = pd.DataFrame(index=keys_k)
    hist4_req_k['h4'] = sinp.general['i_history4_req'] #add the h4 key
    hist4_req_k['req'] = 1 #add the param value +1 which is the require value in pyomo
    hist4_req_r = pd.merge(phases_rotn_df, hist4_req_k, how='left', left_on=sinp.end_col(), right_index=True)
    hist4_req_r = hist4_req_r.drop(list(range(sinp.general['phase_len'])), axis=1)  # drop the segregated landuse cols
    hist4_req_rh4 = hist4_req_r.set_index(['h4'], append=True)
    params['hist4_req'] = hist4_req_rh4.squeeze().to_dict()

    ##hist4 prov - this is just the current landuse
    hist4_prov_r = phases_rotn_df.iloc[:, -1:]
    hist4_prov_r = hist4_prov_r.assign(prov= 1)  # add the param value +1 which is the require value in pyomo
    hist4_prov_rh4 = hist4_prov_r.set_index(hist4_prov_r.columns[0], append=True) #add landuse as index level
    params['hist4_prov'] = hist4_prov_rh4.squeeze().to_dict()

