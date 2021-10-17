##import python modules
import numpy as np

##import AFO module
import PropertyInputs as pinp
import Functions as fun
import SeasonalFunctions as zfun
import Periods as per

na = np.newaxis

def f_season_precalcs(params, r_vals):
    ################
    #z8z9 transfer #
    ################
    ##get param
    date_season_node_p7z = per.f_season_periods()[:-1,...]
    season_start_z = per.f_season_periods()[0,:] #slice season node to get season start
    period_is_seasonstart_p7z = date_season_node_p7z==season_start_z
    mask_provwithinz8z9_p7z8z9, mask_provbetweenz8z9_p7z8z9, mask_reqwithinz8_p7z8, mask_reqbetweenz8_p7z8 = zfun.f_season_transfer_mask(
        date_season_node_p7z, period_is_seasonstart_pz=period_is_seasonstart_p7z, z_pos=-1)  # slice off end date p7

    ###########
    #sequence #
    ###########
    ##lengths
    bool_steady_state = pinp.general['steady_state'] or np.count_nonzero(pinp.general['i_mask_z']) == 1
    if bool_steady_state:
        len_z = 1
    else:
        len_z = np.count_nonzero(pinp.general['i_mask_z'])
    len_q = pinp.general['i_len_q'] #length of season sequence
    len_s = np.power(len_z,len_q - 1)

    ##indexs
    index_q = np.arange(len_q)
    index_s = np.arange(len_s)
    index_z = np.arange(len_z)

    ##season prob
    i_season_propn_z = zfun.f_z_prob()

    ##sequence mask
    ###The number of ‘s’ that are grouped together for each ‘q’
    step_sparam_q = np.power(float(len_z),len_q - 1 - index_q)  # needs to be float
    ###The number of ‘z’ that are grouped together for each ‘q’
    step_zparam_q = np.power(float(len_z),len_q - 2 - index_q)  # needs to be float
    ###Only the first ‘s’ in the group is active (not masked)
    mask_s8vars_qs8 = (index_s % step_sparam_q[:,na] == 0)

    mask_s9vars_qs9 = (index_s % np.roll(step_sparam_q[:,na],-1,axis=0) == 0)
    ## mask prov has 4 parts
    ### 1. the sequences s8 and s9 exist
    ### 2. The sequences s8 & s9 are both in the same 's' step (as determined by trunc() of the step)
    ### 3. The season type within s8 aligns with the position of s9 in the sequence step
    ### 4. All seasons in the sequences in q[-1] pass to q[0]
    mask_provqs8z8s9_qs8z8s9 = (mask_s8vars_qs8[:,:,na,na] * mask_s9vars_qs9[:,na,na,:]
                                * (np.trunc(index_s[:,na,na] / step_sparam_q)
                                   == np.trunc(index_s / step_sparam_q))
                                * (index_z[:,na] == np.trunc(index_s / step_zparam_q) % len_z))
    mask_provqs8z8s9_qs8z8s9[-1, ...] = True

    # mask_provqs8z8s9_qs8z8s9 = (mask_s8vars_qs8[:,:,na,na]
    #                             *(np.trunc(index_s[:,na,na] / step_sparam_q[:,na,na,na]) == np.trunc(index_s / step_sparam_q[:,na,na,na]))
    #                             *(index_z[:,na] == np.trunc(index_s / step_zparam_q[:,na,na,na]) % len_z))

    season_seq_prob_qzs9 = np.cumprod(np.sum(i_season_propn_z[:,na] * mask_provqs8z8s9_qs8z8s9,axis=1,keepdims=False),axis=0)
    season_seq_prob_qsz = np.moveaxis(season_seq_prob_qzs9,source=-1,destination=1)  # s9 to s8 so that s9 no longer exists
    # season_seq_prob_qsz = np.cumprod(np.sum(i_season_propn_z * mask_s8vars_qs8[:,:,na],axis=-1, keepdims=True),
    #                                  axis=0)  # todo as above, work needed to represent sequence of interest. Currently z axis is not active.
    p_wyear_inc_qs = mask_s8vars_qs8  # todo work needed to allow masking ‘sequence of interest’ (which requires a z8 axis).
    p_season_prob_qsz = season_seq_prob_qsz

    p_between_req_qs = mask_s8vars_qs8  # todo work needed to represent the multi-period which requires the final year to be the equilibrium year
    p_sequence_prov_qs8zs9 = mask_s8vars_qs8[:,:,na,na] * (index_q[:,na,na,na] != (len_q - 1)) * mask_provqs8z8s9_qs8z8s9
    p_endstart_prov_qsz = mask_s8vars_qs8[:,:,na] * (index_q[:,na,na] == (len_q - 1)) * season_seq_prob_qsz

    ##########
    #params  #
    ##########
    keys_q = np.array(['q%s' % i for i in range(len_q)])
    keys_s = np.array(['s%s' % i for i in range(len_s)])
    keys_z = zfun.f_keys_z()
    keys_p7 = per.f_season_periods(keys=True)

    ###p7z8
    arrays = [keys_p7, keys_z]
    index_p7z8 = fun.cartesian_product_simple_transpose(arrays)
    tup_p7z8 = tuple(map(tuple,index_p7z8))
    ###p7z8z9
    arrays = [keys_p7, keys_z, keys_z]
    index_p7z8z9 = fun.cartesian_product_simple_transpose(arrays)
    tup_p7z8z9 = tuple(map(tuple,index_p7z8z9))
    ###qs - season sequence
    arrays = [keys_q, keys_s]
    index_qs = fun.cartesian_product_simple_transpose(arrays)
    tup_qs = tuple(map(tuple, index_qs))
    ###qsz - season sequence
    arrays = [keys_q, keys_s, keys_z]
    index_qsz = fun.cartesian_product_simple_transpose(arrays)
    tup_qsz = tuple(map(tuple,index_qsz))
    ###qs8zs9 - season sequence
    arrays = [keys_q, keys_s, keys_z, keys_s]
    index_qs8zs9 = fun.cartesian_product_simple_transpose(arrays)
    tup_qs8zs9 = tuple(map(tuple,index_qs8zs9))


    params['p_childz_reqwithin_season'] =dict(zip(tup_p7z8, mask_reqwithinz8_p7z8.ravel()*1))
    params['p_childz_reqbetween_season'] =dict(zip(tup_p7z8, mask_reqbetweenz8_p7z8.ravel()*1))
    params['p_parentz_provwithin_season'] =dict(zip(tup_p7z8z9, mask_provwithinz8z9_p7z8z9.ravel()*1))
    params['p_parentz_provbetween_season'] =dict(zip(tup_p7z8z9, mask_provbetweenz8z9_p7z8z9.ravel()*1))
    params['p_wyear_inc_qs'] =dict(zip(tup_qs,p_wyear_inc_qs.ravel()*1))
    params['p_between_req_qs'] = dict(zip(tup_qs,p_between_req_qs.ravel()*1))
    params['p_season_prob_qsz'] = dict(zip(tup_qsz,p_season_prob_qsz.ravel()))
    params['p_endstart_prov_qsz'] = dict(zip(tup_qsz,p_endstart_prov_qsz.ravel()))
    params['p_sequence_prov_qs8zs9'] = dict(zip(tup_qs8zs9,p_sequence_prov_qs8zs9.ravel()*1))


    ##report
    r_vals['keys_q'] = keys_q
    r_vals['keys_s'] = keys_s
    r_vals['keys_z'] = keys_z
    r_vals['z_prob_qsz'] = p_season_prob_qsz