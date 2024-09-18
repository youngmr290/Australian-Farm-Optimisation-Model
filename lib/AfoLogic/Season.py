##import python modules
import numpy as np

##import AFO module
from . import PropertyInputs as pinp
from . import UniversalInputs as uinp
from . import StructuralInputs as sinp
from . import Functions as fun
from . import SeasonalFunctions as zfun
from . import Periods as per

na = np.newaxis

def f_season_precalcs(params, r_vals):
    ################
    #p6z9 mask     #
    ################
    ##used to skip constraints
    feed_period_dates_p6z = per.f_feed_periods()
    date_start_p6z = feed_period_dates_p6z[:-1]
    mask_fp_z8var_p6z = zfun.f_season_transfer_mask(date_start_p6z, z_pos=-1, mask=True)


    ################
    #z8z9 transfer #
    ################
    ##get param
    date_season_node_p7z = per.f_season_periods()[:-1,...]
    season_start_z = per.f_season_periods()[0,:] #slice season node to get season start
    period_is_seasonstart_p7z = date_season_node_p7z==season_start_z
    mask_provwithinz8z9_p7z8z9, mask_provbetweenz8z9_p7z8z9, mask_reqwithinz8_p7z8, mask_reqbetweenz8_p7z8 = zfun.f_season_transfer_mask(
        date_season_node_p7z, period_is_seasonstart_pz=period_is_seasonstart_p7z, z_pos=-1)  # slice off end date p7
    mask_season_p7z = zfun.f_season_transfer_mask(date_season_node_p7z,z_pos=-1,mask=True)

    ###########
    #sequence #
    ###########
    ##lengths
    bool_steady_state = sinp.structuralsa['steady_state'] or np.count_nonzero(pinp.general['i_mask_z']) == 1
    if bool_steady_state:
        len_z = 1
    else:
        len_z = np.count_nonzero(pinp.general['i_mask_z'])
    len_q = sinp.structuralsa['i_len_q'] #length of season sequence
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

    ##probability of a weather-year in each year of the sequence (which is the cum prob for year 0) for each p7.
    ## in p7 when the seasons are clustered the parent season probability includes the cluster children.
    len_p7 = date_season_node_p7z.shape[0]
    season_prob_p7z = np.zeros((len_p7, len_z))
    season_prob_p7z[-1,...] = i_season_propn_z
    for p7 in reversed(range(len_p7-1)):
        season_prob_p7z[p7,...] = np.sum(season_prob_p7z[p7+1,na,:] * mask_provwithinz8z9_p7z8z9[p7], axis=-1)
    season_prob_p7z = season_prob_p7z / np.sum(season_prob_p7z, axis=-1, keepdims=True) #scale so prob equals 1
    prob_qszp7 = mask_s8vars_qs8[..., na, na] * season_prob_p7z.T
    season_seq_prob_qszp7 = prob_qszp7.copy()
    for q in range(1, len_q):   #calculate cum prob in a loop because requires summing across z in q_prev
        season_seq_prob_qszp7[q, ...] = prob_qszp7[q,...] * np.sum(season_seq_prob_qsz[q-1, :, :, na]
                                                                * parent_qs9_qs8zs9[q,...], axis = (0,1))[:,na,na]

    p_wyear_inc_qs = mask_s8vars_qs8  # todo work needed to allow masking ‘sequence of interest’ (which requires a z8 axis).
    p_season_prob_qsz = season_seq_prob_qsz / len_q # Divide by len_q so that the objective value is $/yr rather than $/sequence

    ##alter the probability to represent a 10yr planning horizon and add discount it to count for time value of money
    ## dont need to adjust season_seq_prob_qszp7 because it doesnt get divided by len_q (it only gets used in bnds)
    if sinp.structuralsa['model_is_MP']:
        ###if len_q is less than the length of the planning horizon then add the probabilities to the final year
        len_planning_horizon = 10
        planning_years_per_q = np.ones(len_q)
        planning_years_per_q[-1] = planning_years_per_q[-1] + len_planning_horizon - len_q
        p_season_prob_qsz = season_seq_prob_qsz / len_planning_horizon * planning_years_per_q[:,na,na]

        ##calc discount factor each year over the planning horizon
        discount_rate = uinp.finance['i_interest']
        discount_factor = 1/(1+discount_rate)**np.arange(len_planning_horizon)
        ###if len_q is less than the length of the planning horizon then add the probabilities to the final year
        discount_factor_q = discount_factor[0:len_q]
        discount_factor_q[-1] = discount_factor_q[-1] + sum(discount_factor[len_q:])
        discount_factor_q = discount_factor_q/planning_years_per_q
    else:
        discount_factor_q = np.ones(len_q) #only use a discount factor if MP model.

    p_sequence_prov_qs8zs9 = mask_s8vars_qs8[:,:,na,na] * (index_q[:,na,na,na] != (len_q - 1)) * mask_provqs8z8s9_qs8z8s9
    p_endstart_prov_qsz = mask_s8vars_qs8[:,:,na] * (index_q[:,na,na] == (len_q - 1)) * season_seq_prob_qsz

    ##########
    #params  #
    ##########
    keys_q = np.array(['q%s' % i for i in range(len_q)])
    keys_s = np.array(['s%s' % i for i in range(len_s)])
    keys_z = zfun.f_keys_z()
    keys_p6 = np.asarray(pinp.period['i_fp_idx'])
    keys_p7 = per.f_season_periods(keys=True)

    ###p6z
    arrays_p6z = [keys_p6, keys_z]
    ###p7z8
    arrays_p7z8 = [keys_p7, keys_z]
    ###p7z8z9
    arrays_p7z8z9 = [keys_p7, keys_z, keys_z]
    ###qs - season sequence
    arrays_qs = [keys_q, keys_s]
    ###qsz - season sequence
    arrays_qsz = [keys_q, keys_s, keys_z]
    arrays_qszp7 = [keys_q, keys_s, keys_z, keys_p7]
    ###qs8zs9 - season sequence
    arrays_qs8zs9 = [keys_q, keys_s, keys_z, keys_s]

    params['p_mask_fp_z8var_p6z'] = fun.f1_make_pyomo_dict(mask_fp_z8var_p6z * 1, arrays_p6z)
    params['p_mask_season_p7z'] = fun.f1_make_pyomo_dict(mask_season_p7z * 1, arrays_p7z8)
    params['p_mask_childz_within_season'] = fun.f1_make_pyomo_dict(mask_reqwithinz8_p7z8*1, arrays_p7z8)
    params['p_mask_childz_between_season'] = fun.f1_make_pyomo_dict(mask_reqbetweenz8_p7z8*1, arrays_p7z8)
    params['p_parentz_provwithin_season'] = fun.f1_make_pyomo_dict(mask_provwithinz8z9_p7z8z9*1, arrays_p7z8z9)
    params['p_parentz_provbetween_season'] = fun.f1_make_pyomo_dict(mask_provbetweenz8z9_p7z8z9*1, arrays_p7z8z9)
    params['p_wyear_inc_qs'] = fun.f1_make_pyomo_dict(p_wyear_inc_qs*1, arrays_qs)
    params['p_discount_factor_q'] =  dict(zip(keys_q, discount_factor_q))
    params['p_season_prob_qsz'] = fun.f1_make_pyomo_dict(p_season_prob_qsz, arrays_qsz)
    params['p_season_seq_prob_qszp7'] = fun.f1_make_pyomo_dict(season_seq_prob_qszp7, arrays_qszp7)
    params['p_endstart_prov_qsz'] = fun.f1_make_pyomo_dict(p_endstart_prov_qsz, arrays_qsz)
    params['p_sequence_prov_qs8zs9'] = fun.f1_make_pyomo_dict(p_sequence_prov_qs8zs9*1, arrays_qs8zs9)

    ##store r_vals
    fun.f1_make_r_val(r_vals,keys_q,'keys_q')
    fun.f1_make_r_val(r_vals,keys_s,'keys_s')
    fun.f1_make_r_val(r_vals,keys_z,'keys_z')
    fun.f1_make_r_val(r_vals,keys_p7,'keys_p7')
    fun.f1_make_r_val(r_vals,discount_factor_q,'discount_factor_q')
    fun.f1_make_r_val(r_vals,p_season_prob_qsz,'z_prob_qsz')
    fun.f1_make_r_val(r_vals,mask_season_p7z,'mask_season_p7z')
    fun.f1_make_r_val(r_vals,p_wyear_inc_qs,'mask_qs')
    fun.f1_make_r_val(r_vals,date_season_node_p7z % 364,'date_season_node_p7z') #mod 364 so that all dates are from the start of the yr (makes it easier to compare in the report)