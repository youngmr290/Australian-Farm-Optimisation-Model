"""

This module reads in rotation information that has been generated from RotGeneration.py and manipulates
it to produce the rotation parameters.

author: young
"""

#python modules
import pandas as pd
import numpy as np
na = np.newaxis
import timeit

#AFO modules - a bunch of other precalc modules import this module
import StructuralInputs as sinp
import PropertyInputs as pinp
import Functions as fun
import Periods as per
import SeasonalFunctions as zfun


def f_v_phase_increment_adj(param, p7_pos, z_pos, p5_pos=None, numpy=False):
    '''
    Adjust v_phase param for v_phase_change_increase.

    v_phase_change_increase must incur the requirement to date for labour and cash for the phase.
    This is making the assumption that any jobs carried out, and any expenditure
    (fertiliser or chemical applied) will be applied even though the phase is selected later in the year.
    This stops the optimisation selecting the phase in the last node and receiving the income without
    incurring any costs. Note: Yield and stubble do not require increment params because it is not possible to harvest a
    rotation before the rotation is selected.

    Note labour gets handled slightly different. Labour that occurs in previous labour periods before the season
        period when v_phase_change_increase is selected must be completed in the first labour period when the phase is selected.

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
        ##get p5 periods into corect shape
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
    phases_df = pinp.f1_phases()
    landuse_r = phases_df.iloc[:,-1].values
    dry_sown_landuses = sinp.landuse['dry_sown']
    phase_is_drysown_r = np.any(landuse_r[:,na]==list(dry_sown_landuses), axis=-1)

    ##z8z9 transfer
    start_phase_periods_p7z = per.f_season_periods()[:-1,:] #remove end date of last period
    season_start_z = per.f_season_periods()[0,:] #slice season node to get season start
    period_is_seasonstart_p7z = start_phase_periods_p7z==season_start_z
    mask_provwithinz8z9_p7z8z9, mask_provbetweenz8z9_p7z8z9, mask_reqwithinz8_p7z8, mask_reqbetweenz8_p7z8 = zfun.f_season_transfer_mask(
        start_phase_periods_p7z, period_is_seasonstart_pz=period_is_seasonstart_p7z, z_pos=-1) #the req masks don't do the correct job for rotation and hence are not used.
    ###for rotation the between and within constraints are acting on different things (history vs the acutal phase) therefore
    ### the req params above don't work because they have been adjusted for season start. so in the following line i make a
    ### new req param which doesn't account for within or between
    mask_childz8_p7z8 = zfun.f_season_transfer_mask(start_phase_periods_p7z,z_pos=-1,mask=True)

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

    params['p_mask_childz_phase'] = fun.f1_make_pyomo_dict(mask_childz8_p7z8*1, arrays_p7z8)
    params['p_parentz_provwithin_phase'] = fun.f1_make_pyomo_dict(mask_provwithinz8z9_p7z8z9*1, arrays_p7z8z9)
    params['p_parentz_provbetween_phase'] = fun.f1_make_pyomo_dict(mask_provbetweenz8z9_p7z8z9*1, arrays_p7z8z9)
    # params['p_mask_phases'] =dict(zip(tup_rm, mask_phases_rm.ravel()*1))
    # params['p_dryz_link'] =dict(zip(tup_rz8z9, mask_drynext_z8z9.ravel()*1))
    # params['p_dryz_link2'] =dict(zip(tup_rz8z9, mask_drystart_z8z9.ravel()*1))


def f_landuses_phases(params,r_vals):
    '''
    * Read in the rotation list generated by RotGeneration.py
    * Create rotation area parameter for pyomo (simply each rotation phase uses 1ha of area).
    * Store rotation list and pasture phases list to report dictionary

    '''
    phases=pinp.f1_phases()
    phases_rk = phases.set_index(phases.columns[-1], append=True) #add landuse as index level
    params['phases_rk'] = dict.fromkeys(phases_rk.index,1)

    ##store r_vals
    fun.f1_make_r_val(r_vals,phases,'phases')
    fun.f1_make_r_val(r_vals,sinp.landuse['All_pas'],'all_pastures')#all_pas2 includes the cont pasture landuses


def f_rot_lmu_params(params):
    '''
    Create parameters for lmu area.

    '''
    ##area
    lmu_mask = pinp.general['i_lmu_area'] > 0
    params['lmu_area'] = dict(zip(pinp.general['i_lmu_idx'][lmu_mask], pinp.general['i_lmu_area'][lmu_mask]))


def f_rot_hist_params(params):
    '''
    Create parameters for landuse history provided and required by each rotation phase.

    '''
    rot_req = pd.read_excel('Rotation.xlsx', sheet_name='rotation_req', header= None, engine='openpyxl')#, index_col = [0,1]) #couldn't get it to read in with multiindex for some reason
    rot_prov = pd.read_excel('Rotation.xlsx', sheet_name='rotation_prov', header= None, engine='openpyxl')#, index_col = [0,1]) #couldn't get it to read in with multiindex for some reason
    rot_req = rot_req.set_index([0,1])
    rot_prov = rot_prov.set_index([0,1])
    params['hist_prov'] = rot_prov.squeeze().to_dict()
    params['hist_req'] = rot_req.squeeze().to_dict()



