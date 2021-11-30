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
                                * (np.trunc(index_s[:,na,na] / step_sparam_q[:,na,na,na])
                                   == np.trunc(index_s / step_sparam_q[:,na,na,na]))
                                * (index_z[:,na] == np.trunc(index_s / step_zparam_q[:,na,na,na]) % len_z))
    mask_provqs8z8s9_qs8z8s9[-1, ...] = False  #in the final year of the sequence all the weather years only 'provide' to the initial sequence (s9[0])
    mask_provqs8z8s9_qs8z8s9[-1, ..., 0] = True
    parent_qs9_qs8zs9 = np.roll(mask_provqs8z8s9_qs8z8s9, 1, axis = 0)  #the parent of q,s9 is q_prev,s8,z

    ##probability of a weather-year in each year of the sequence (which is the cum prob for year 0)
    ###This is the place to alter probability if weather-years are not independent (by not broadcasting across q)
    prob_qsz = mask_s8vars_qs8[..., na] * i_season_propn_z
    season_seq_prob_qsz = prob_qsz.copy()
    for q in range(1, len_q):   #calculate cum prob in a loop because requires summing across z in q_prev
        season_seq_prob_qsz[q, ...] = prob_qsz[q, ...] * np.sum(season_seq_prob_qsz[q-1, :, :, na]
                                                                * parent_qs9_qs8zs9[q, ...], axis = (0,1))[...,na]

    p_wyear_inc_qs = mask_s8vars_qs8  # todo work needed to allow masking ‘sequence of interest’ (which requires a z8 axis).
    p_season_prob_qsz = season_seq_prob_qsz / len_q # Divide by len_q so that the objective value is $/yr rather than $/sequence

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
    arrays_p7z8 = [keys_p7, keys_z]
    ###p7z8z9
    arrays_p7z8z9 = [keys_p7, keys_z, keys_z]
    ###qs - season sequence
    arrays_qs = [keys_q, keys_s]
    ###qsz - season sequence
    arrays_qsz = [keys_q, keys_s, keys_z]
    ###qs8zs9 - season sequence
    arrays_qs8zs9 = [keys_q, keys_s, keys_z, keys_s]


    params['p_mask_childz_within_season'] = fun.f1_make_pyomo_dict(mask_reqwithinz8_p7z8*1, arrays_p7z8)
    params['p_mask_childz_between_season'] = fun.f1_make_pyomo_dict(mask_reqbetweenz8_p7z8*1, arrays_p7z8)
    params['p_parentz_provwithin_season'] = fun.f1_make_pyomo_dict(mask_provwithinz8z9_p7z8z9*1, arrays_p7z8z9)
    params['p_parentz_provbetween_season'] = fun.f1_make_pyomo_dict(mask_provbetweenz8z9_p7z8z9*1, arrays_p7z8z9)
    params['p_wyear_inc_qs'] = fun.f1_make_pyomo_dict(p_wyear_inc_qs*1, arrays_qs)
    params['p_season_prob_qsz'] = fun.f1_make_pyomo_dict(p_season_prob_qsz, arrays_qsz)
    params['p_endstart_prov_qsz'] = fun.f1_make_pyomo_dict(p_endstart_prov_qsz, arrays_qsz)
    params['p_sequence_prov_qs8zs9'] = fun.f1_make_pyomo_dict(p_sequence_prov_qs8zs9*1, arrays_qs8zs9)


    ##report
    r_vals['keys_q'] = keys_q
    r_vals['keys_s'] = keys_s
    r_vals['keys_z'] = keys_z
    r_vals['z_prob_qsz'] = p_season_prob_qsz