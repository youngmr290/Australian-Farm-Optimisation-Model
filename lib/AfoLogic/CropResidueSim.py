"""

The AFO residue simulator estimates the proportion of the crop residue in each category using trial data of
sequential animal liveweights grazing crop residue of a crop with known yield. There are 2 steps to using
the residue simulator:

    1.	The discrete animal liveweights (from trial data) are converted to a continuous function by fitting
        a quadratic function to the liveweights and the number of grazing days since grazing started
        (LW = a GD2 + b GD + c). This approach is as developed by Thomas et al. 2021. Using grazing days
        as the independent variable allows trials that had varying stock numbers to be included.

        The fitted quadratic curves are a better comparison of the grazing achieved from each crop or
        site than liveweight versus date, because it includes the impact of stock numbers. However, it
        is not accounting for the differences in the yield of the crops between sites. Yield is
        accounted for in step 2.

    2.	Run the AFO residue simulator (CropResidueSim.py) which uses inputs from Property.xlsx. The inputs include

        a.	Crop yield
        b.	Coefficients from the quadratic LW function
        c.	The commencement day of the trial after the harvest date, so that feed deterioration can be estimated.
        d.	Description of the animals that were grazing in the trial, including genotype (SRW), age, fleece length (for insulation), the physiological state of ewes (pregnant or lactating with litter size)
        e.	Description of the environment for calculation of chill effects (wind speed, rainfall and temperature).
        f.	Level of supplement consumed.

As part of the Crop Residue Simulator, the stock generator (see :ref:`livestock_ref` for more information) is run for a range of
feed qualities beginning at the week corresponding to the start of grazing in the trial. The liveweight
change (LWC), metabolizable intake (MEI) & dry matter intake (DMI) is recorded for each feed quality
assessed. The LWC of the animals in the trial, as estimated from the fitted quadratic, is compared to
the range of feed qualities assessed to estimate the quality and quantity of the diet that was consumed
by the animals in the trial. These calculations are carried out for a sufficient period that
stubble (or fodder crop) from all the AFO stubble categories will have been grazed. The fitting
of the quadratic to the measured LW means that the calculation of intake can be extended beyond
the duration of the measurements made in the trial. The quantity of each stubble category is then
calculated as a proportion of the total residue mass at harvest, and these values are the inputs
stored in stubble.xlsx.

Multiple paddock trials that can be used to generate inputs for a single crop. If multiple trials are activated
the results (proportion of residue in each category) is averaged. This is controlled in the inputs.

The module writes the answers to an excel book which is referenced by AFO - this means this module only needs to be run
if you make changes to stub inputs or if you change the inputs/formulas that calculate lwc in the generator (it could be
made so that the stubble generator is run for every trial but this will slow AFO and potentially confuse the interpretation
of results).

27/2/2022: currently there is only liveweight data for GSW region. Thus CWW uses the same stubble. This would be improved
by running a stubble trial in the wheatbelt and recording liveweight every week.

@author: young
"""
#python modules
import pandas as pd
import numpy as np
from scipy.optimize import minimize

##do this so the module doesnt run when building the docs
if __name__=="__main__":
    #AFO modules
    from lib.RawVersion import LoadExcelInputs as dxl
    from lib.RawVersion import LoadExp as exp
    from lib.RawVersion import RawVersionExtras as rve
    from lib.AfoLogic import UniversalInputs as uinp
    from lib.AfoLogic import PropertyInputs as pinp
    from lib.AfoLogic import StructuralInputs as sinp
    from lib.AfoLogic import StockFunctions as sfun
    from lib.AfoLogic import Functions as fun
    from lib.AfoLogic import Sensitivity as sen
    from lib.AfoLogic import StockGenerator as sgen
    from lib.AfoLogic import CropResidue as stub
    from lib.AfoLogic import relativeFile

    na = np.newaxis

    #todo currently to handle multiple trials we need to run the simulator multiple time. A good long term solution will be
    # to add trial as an axis. This shouldn't be too hard - just need to expanded the named ranges in pinp.xl and add axis below.

    ###############
    #User control #
    ###############
    trial = 31   #Count the number of rows (starting at 0) offset from the Default trial

    ######
    #Run #
    ######
    ##load excel data and experiment data
    exp_data, exp_group_bool, trial_pinp = exp.f_read_exp()
    sinp_defaults, uinp_defaults, pinp_defaults = dxl.f_load_excel_default_inputs(trial_pinp=trial_pinp)
    d_rot_info = dxl.f_load_phases()
    cat_propn_s1_ks2 = dxl.f_load_stubble()

    ##select property for the current trial
    property = trial_pinp.iloc[trial]

    ##process user SA
    user_sa = rve.f_process_user_sa(exp_data, trial)

    ##select property and reset default inputs for the current trial. Must occur first.
    sinp.f_select_n_reset_sinp(sinp_defaults)
    sinp.f_landuse_sets()
    uinp.f_select_n_reset_uinp(uinp_defaults)
    pinp.f_select_n_reset_pinp(property, pinp_defaults)

    ##update sensitivity values
    sen.create_sa()
    fun.f_update_sen(user_sa,sen.sam,sen.saa,sen.sap,sen.sar,sen.sat,sen.sav)

    ##call sa functions - assigns sa variables to relevant inputs
    sinp.f_structural_inp_sa(sinp_defaults)
    uinp.f_universal_inp_sa(uinp_defaults)
    pinp.f_property_inp_sa(pinp_defaults)

    ##expand p6 axis to include nodes
    sinp.f1_expand_p6()
    pinp.f1_expand_p6()

    #################
    #inputs for sgen#
    #################
    ##inputs are stored in a dict
    stubble_inp = {}
    ##create fs - read from inputs

    ##sim run periods - start and end p
    trial_commencement_date_t = pinp.stubble['start_trial_t']
    n_sim_periods, date_start_p, date_start_P, date_end_p, date_end_P, p_index_p, step \
        = sfun.f1_sim_periods(sinp.stock['i_sim_periods_year'], sinp.stock['i_age_max'], pinp.sheep['i_o_len'])
    n_sim_periods_offs, offs_date_start_p, offs_date_start_P, offs_date_end_p, offs_date_end_P, p_index_offs_p, step \
        = sfun.f1_sim_periods(sinp.stock['i_sim_periods_year'], sinp.stock['i_age_max_offs'], pinp.sheep['i_o_len'])
    mask_p_offs_p = p_index_p<=(n_sim_periods_offs-1)

    ###scale trial start to the correct yr in the sim based on animal age
    add_yrs_t = np.ceil((date_start_p[0] - trial_commencement_date_t) / 364)
    # sub_yrs = np.ceil(np.maximum(0, (item_start - end_of_periods).astype('timedelta64[D]').astype(int) / 365))
    trial_commencement_date_t = trial_commencement_date_t + add_yrs_t * 364
    ####scale for animal age
    trial_commencement_date_t = trial_commencement_date_t + pinp.stubble['animal_age_t'] * 364

    ##general info
    b0_pos = sinp.stock['i_b0_pos']
    b1_pos = sinp.stock['i_b1_pos']
    p_pos = sinp.stock['i_p_pos']
    s1_pos = sinp.stock['i_w_pos'] #s1 goes in w pos for the stubble sim

    len_k = len(sinp.landuse['C'])
    len_s2 = len(pinp.stubble['i_idx_s2'])
    len_p1 = n_sim_periods
    len_s1 = len(pinp.stubble['i_stub_cat_dmd_s1'])

    ##read in and reshape the array that specifies which trial/s are used for each crop.
    trial_inc_tks2 = pinp.stubble['i_t_inc_tks2'].reshape(-1, len_k, len_s2)
    n_trials = trial_inc_tks2.shape[0]

    ##determine sgen run periods
    p_start_trial_t = np.searchsorted(date_start_p, trial_commencement_date_t)
    p_start_harv_t = np.searchsorted(date_start_p, trial_commencement_date_t - pinp.stubble['i_calibration_offest_t'])
    p_end_t = p_start_trial_t + pinp.stubble['trial_length_t']

    ##lw each period - based on the fitted quadratic on grazing days (gdays) from the paddock data.
    stocking_rate_ts2 = pinp.stubble['i_sr_s2t'].T
    gdays_since_trialstart_tps2 = ((date_start_p - date_start_p[p_start_trial_t[:, na]])[:, :, na]
                                  * stocking_rate_ts2[:,na,:] / 100).astype(int)  # grazing days (100s) since trial start
    a_tks2 = pinp.stubble['i_a_tks2'].astype(float).reshape(-1,len_k, len_s2)
    b_tks2 = pinp.stubble['i_b_tks2'].astype(float).reshape(-1,len_k, len_s2)
    c_tks2 = pinp.stubble['i_c_tks2'].astype(float).reshape(-1,len_k, len_s2)
    trial_lw_tpks2 = a_tks2[:,na,:,:] * gdays_since_trialstart_tps2[:, :, na, :] ** 2 + b_tks2[:,na,:,:] * gdays_since_trialstart_tps2[:, :, na, :] + c_tks2[:,na,:,:]

    ##yield from trial
    trial_yield_tk = pinp.stubble['i_trial_yield_tk'].astype(float).reshape(-1,len_k)

    ##dmd categories to generate - include deterioration.
    ## deterioration is since harvest because the definition of the categories are at harvest.
    dmd_s1 = pinp.stubble['i_stub_cat_dmd_s1'] #at harvest
    days_since_harv_tp = (date_start_p - date_start_p[p_start_harv_t[:,na]]).astype(int) #days since harvest
    dmd_tps1k = dmd_s1[:,na] * (1 - pinp.stubble['quality_deterioration']) ** days_since_harv_tp[:,:,na,na]

    ##initilise arrays so they can be assigned by k
    lwc_p1s1ks2 = np.zeros((len_p1,len_s1,len_k,len_s2))
    intake_p1s1ks2 = np.zeros((len_p1,len_s1,len_k,len_s2))
    cat_propn_ts1ks2 = np.zeros((n_trials,len_s1,len_k,len_s2))

    for t in range(n_trials):
        for k in range(len_k):
            for s2 in range(len_s2):
                if not trial_inc_tks2[t,k,s2]:
                    continue

                ##call stock gen
                stubble_inp['shear_date'] = pinp.stubble['shear_date_t'][t]
                stubble_inp['lambing_date'] = pinp.stubble['lambing_date_t'][t]
                stubble_inp['a_c2_c0'] = pinp.stubble['a_c2_c0t'][:,t]
                stubble_inp['i_g3_inc'] = pinp.stubble['i_g3_inc_g3t'][:,t]
                stubble_inp['i_sr'] = pinp.stubble['i_sr_s2t'][s2,t]
                stubble_inp['i_ws'] = pinp.stubble['i_ws_t'][t]
                stubble_inp['i_rain'] = pinp.stubble['i_rain_t'][t]
                stubble_inp['i_temp_ave'] = pinp.stubble['i_temp_ave_t'][t]
                stubble_inp['i_temp_max'] = pinp.stubble['i_temp_max_t'][t]
                stubble_inp['i_temp_min'] = pinp.stubble['i_temp_min_t'][t]
                stubble_inp['i_gfw'] = pinp.stubble['i_gfw_t'][t]
                stubble_inp['i_fd'] = pinp.stubble['i_fd_t'][t]
                stubble_inp['i_fl'] = pinp.stubble['i_fl_t'][t]
                stubble_inp['i_md'] = pinp.stubble['i_md_t'][t]
                stubble_inp['w_foetus_start'] = pinp.stubble['w_foetus_start_t'][t]
                stubble_inp['i_lw_yatf'] = pinp.stubble['i_lw_yatf_t'][t]
                stubble_inp['i_gfw_yatf'] = pinp.stubble['i_gfw_yatf_t'][t]
                stubble_inp['i_fl_yatf'] = pinp.stubble['i_fl_yatf_t'][t]
                stubble_inp['i_fd_yatf'] = pinp.stubble['i_fd_yatf_t'][t]
                stubble_inp['i_fat_yatf'] = pinp.stubble['i_fat_yatf_t'][t]
                stubble_inp['i_muscle_yatf'] = pinp.stubble['i_muscle_yatf_t'][t]
                stubble_inp['i_viscera_yatf'] = pinp.stubble['i_viscera_yatf_t'][t]
                stubble_inp['i_foo'] = pinp.stubble['i_foo_t'][t]
                stubble_inp['i_sup_intake'] = pinp.stubble['i_sup_intake_t'][t]
                stubble_inp['p_start'] = p_start_trial_t[t]
                stubble_inp['p_end'] = p_end_t[t]
                stubble_inp['lw'] = trial_lw_tpks2[t,:,k,s2]
                stubble_inp['dmd_pw'] = dmd_tps1k[t,:,:,k]
                o_stub_intake_tpdams, o_stub_intake_tpoffs, o_ebg_tpdams, o_ebg_tpoffs = sgen.generator(stubble=stubble_inp)

                ##slice based on animal in trial
                ## currently only the g and b axis are selected based on trial info. Any other axes are averaged. (This could be changed).
                if pinp.stubble['i_dams_in_trial_t'][t]:
                    ###select across g axis - weighted
                    mask_dams_inc_g1 = np.any(sinp.stock['i_mask_g1g3'] * pinp.sheep['i_g3_inc'], axis=1)
                    mask_offs_inc_g3 = np.any(sinp.stock['i_mask_g3g3'] * pinp.sheep['i_g3_inc'], axis=1)
                    o_ebg_tpdams = np.compress(mask_dams_inc_g1, o_ebg_tpdams, axis=-1)
                    o_stub_intake_tpdams = np.compress(mask_dams_inc_g1, o_stub_intake_tpdams, axis=-1)
                    ###select across b axis - weighted
                    i_b1_propn_b1g = fun.f_expand(pinp.stubble['i_b1_propn_b1t'][:,t], b1_pos)
                    lwc_ps1g = np.sum(o_ebg_tpdams * i_b1_propn_b1g, b1_pos, keepdims=True)
                    intake_ps1g = np.sum(o_stub_intake_tpdams * i_b1_propn_b1g, b1_pos, keepdims=True)
                    ###average remaining axes
                    lwc_p1s1ks2[:,:,k,s2] = fun.f_reduce_skipfew(np.average, lwc_ps1g, preserveAxis=(p_pos, s1_pos))
                    intake_p1s1ks2[:,:,k,s2] = fun.f_reduce_skipfew(np.average, intake_ps1g, preserveAxis=(p_pos, s1_pos))
                else:
                    ###select across g axis - weighted
                    mask_offs_inc_g3 = np.any(sinp.stock['i_mask_g3g3'] * pinp.sheep['i_g3_inc'], axis=1)
                    o_ebg_tpoffs = np.compress(mask_offs_inc_g3, o_ebg_tpoffs, axis=-1)
                    o_stub_intake_tpoffs = np.compress(mask_offs_inc_g3, o_stub_intake_tpoffs, axis=-1)
                    ###select across b axis - weighted
                    i_b0_propn_b0g = fun.f_expand(pinp.stubble['i_b0_propn_b0t'][:,t], b0_pos)
                    lwc_ps1g = np.sum(o_ebg_tpoffs * i_b0_propn_b0g, b0_pos, keepdims=True)
                    intake_ps1g = np.sum(o_stub_intake_tpoffs * i_b0_propn_b0g, b0_pos, keepdims=True)
                    ###average remaining axes
                    lwc_p1s1ks2[mask_p_offs_p,:,k,s2] = fun.f_reduce_skipfew(np.average, lwc_ps1g, preserveAxis=(p_pos, s1_pos))
                    intake_p1s1ks2[mask_p_offs_p,:,k,s2] = fun.f_reduce_skipfew(np.average, intake_ps1g, preserveAxis=(p_pos, s1_pos))

        ##post process the lwc
        ###calc trial lw with p1p2 axis (p2 axis is days)
        len_p2 = int(step)
        index_p2 = np.arange(len_p2)
        date_start_p1p2 = date_start_p[..., na] + index_p2
        gdays_since_trialstart_p1p2s2 = ((date_start_p1p2 - date_start_p[p_start_trial_t[t],na])[:,:,na]
                                         * stocking_rate_ts2[t,:]/100)  # .astype(int)  # grazing days (100s) since trial start
        # trial_lw_p1p2ks2 = (a_tks2[t,...] * gdays_since_trialstart_p1p2s2[:,:,na,:] ** 2
        #                     + b_tks2[t,...] * gdays_since_trialstart_p1p2s2[:,:,na,:] + c_tks2[t,...])
        # trial_lw_pks2 = trial_lw_p1p2ks2.reshape(-1,len_k,len_s2)
        # trial_lwc_pks2 = np.roll(trial_lw_pks2, shift=-1, axis=0) - trial_lw_pks2
        # trial_lwc_p1p2ks2 = trial_lwc_pks2.reshape(-1,len_p2, len_k, len_s2)
        trial_lwc_p1p2ks2 = (2 * a_tks2[t,...] * gdays_since_trialstart_p1p2s2[:, :, na, :]
                             + b_tks2[t,...]) * stocking_rate_ts2[t, ...] / 100
        ###calc grazing days in generator period for each dmd - allocate trial lwc to the simulated lwc and sum the p2
        lwc_diff_p1p2s1ks2 = np.abs(lwc_p1s1ks2[:,na,:,:,:] - trial_lwc_p1p2ks2[:,:,na,:,:])
        days_grazed_each_cat_p1s1ks2 = np.sum(np.equal(np.min(lwc_diff_p1p2s1ks2, axis=2,keepdims=True) , lwc_diff_p1p2s1ks2), axis=1)
        ###adjust intake - allowing for decay related to quantity (to reflect the amount at harvest). (Trampling done below).
        adj_intake_p1s1ks2 = intake_p1s1ks2 / (1 - pinp.stubble['quantity_decay'][:,na]) ** days_since_harv_tp[t,:, na, na, na]
        ###multiply by adjusted intake and sum p axis to return the total intake for each dmd (stubble) category
        total_intake_s1ks2 = np.sum(days_grazed_each_cat_p1s1ks2 * adj_intake_p1s1ks2, axis=0)
        total_intake_ha_s1ks2 = total_intake_s1ks2 * stocking_rate_ts2[t,:]
        ###adjust for trampling -
        ### Trampling gets added on to reflect the amount of stubble at harvest.
        #todo Trampling should be the % of the quantity consumed spread across the remaining stubble in the proportion that it exists. But that is difficult in the main code, so it is just the the % of the current category for now
        tramp_ks2 = pinp.stubble['trampling'][:,na]
        total_intake_ha_s1ks2 = total_intake_ha_s1ks2 * (1 + tramp_ks2)   #todo 5Mar24 was:    + tramp_ks2 * np.cumsum(total_intake_ha_s1ks2, axis=0)
        ###set a minimum for each category so that the transfer between cats can always occur.
        total_intake_ha_s1ks2 = np.maximum(1, total_intake_ha_s1ks2) #minimum of 1kg in each category so stubble can always be transferred between categories.
        ###divide intake by total stubble to return stubble proportion in each category
        harvest_index_k = pinp.stubble['i_harvest_index_ks2'][:,0] #select the harvest s2 slice because yield penalty is inputted as a harvestable grain
        biomass_k = trial_yield_tk[t,:] / harvest_index_k
        total_residue_ks2 = biomass_k[:,na] * stub.f_biomass2residue(residuesim=True)
        #adjust the total residue so that consumption can't exceed the biomass
        total_residue_ks2 = np.maximum(total_residue_ks2, np.sum(total_intake_ha_s1ks2, axis=0))
        # total_residue_ks2 = 10000     #set to 10000 if the grain yield is to be back calculated
        cat_propn_ts1ks2[t,...] = fun.f_divide(total_intake_ha_s1ks2, total_residue_ks2)

    ##average across t if multiple trials used to generate inputs
    cat_propn_s1ks2 = np.sum(cat_propn_ts1ks2 * (trial_inc_tks2/np.sum(trial_inc_tks2, axis=0))[:,na,...], axis=0)

    # Create a Pandas Excel writer using XlsxWriter as the engine. used to write to multiple sheets in excel
    stubble_sim_path = relativeFile.findExcel('stubble sim.xlsx')
    writer = pd.ExcelWriter(stubble_sim_path, engine='xlsxwriter')
    cat_propn_s1_ks2 = pd.DataFrame(cat_propn_s1ks2.reshape(len_s1,len_k*len_s2))
    cat_propn_s1_ks2.to_excel(writer,index=False,header=False)
    writer.close()



#########
#old sim#
#########
# ##inputs
# hi_k = pinp.stubble['harvest_index']
# index_k = pinp.stubble['i_stub_landuse_idx']
# proportion_grain_harv_k = pinp.stubble['proportion_grain_harv']
# stub_cat_prop_ks1 = pinp.stubble['stub_cat_prop']
#
# ##calc the dmd of each component at the point when category dmd was calibrated
# deterioration_factor_ks0 = pinp.stubble['quality_deterioration']
# days_since_harv = pinp.stubble['i_calibration_offest']
# dmd_component_harv_ks0 = pinp.stubble['component_dmd'] #dmd at harvest
# dmd_component_ks0 = ((1 - deterioration_factor_ks0) ** days_since_harv) * dmd_component_harv_ks0
#
#
# for crp in range(len(index_k)):
#     ######
#     #sim #
#     ######
#
#     def grain_prop():
#         '''calc grain propn in stubble
#
#         HI = total grain / total biomass (total biomass includes grain as well)
#         stubble = leaf and stalk plus split grain
#         '''
#         hi = hi_k[crp]
#         harv_prop = proportion_grain_harv_k[crp]
#         splitgrain_propn_totalbiomass = hi*(1-harv_prop) #split grain as a propn of total biomass
#         leafstalk_propn_totalbiomass = (1-hi) #leaf & stalk as a propn of total biomass
#         stubble_propn_totalbiomass = splitgrain_propn_totalbiomass + leafstalk_propn_totalbiomass #stubble as a propn of total biomass
#         return splitgrain_propn_totalbiomass/stubble_propn_totalbiomass * 100 #split grain as propn of stubble
#
#     #quantity of each stubble component at harvest
#     def stubble_sim(x):
#         #variables to be solved for (this is new to this version of the sim)
#         z,w,q, g, b, s, c = x
#
#
#         component_proportion={'grain' : grain_prop()
#         ,'blade' : z
#         ,'sheath': w
#         ,'chaff' : q}
#
#         #might be worth making this a proper constrain, either make stem a variable then then con is the sum of all == 100
#         component_proportion['stem'] = 100- (z+w+q+grain_prop())
#
#
#         #variables to be solved for (this is new to this version of the sim)
#         grazing_pref_component={'grain' :  g
#         ,'blade' : b
#         ,'sheath': s
#         ,'chaff' : c
#         ,'stem' :1}
#         #sim length
#         sim_length = int(100/pinp.stubble['step_size'])
#         #number of components
#         number_of_components = len(component_proportion)
#         #numpy array for each stubble section used in sim
#         stubble_availability=np.zeros([number_of_components,sim_length])
#         weighted_availability=np.zeros([number_of_components+1,sim_length]) #extra one for a total tally which is required
#         consumption=np.zeros([number_of_components,sim_length])
#         cumulative_consumption=np.zeros([number_of_components,sim_length])
#         #fill in each numpy array one step at a time. have to fill in each step for each array one at a time because the arrays are linked therefore each array used values from another
#         for step in range(sim_length):
#             #stubble availability (at the start of the sim this is component propn it then decreases depending on which components are consumed)
#             for component, proportion,component_num in zip(component_proportion.keys(),component_proportion.values(),range(number_of_components)):
#                 if step == 0:
#                     stubble_availability[component_num, step]=proportion
#                 elif stubble_availability[component_num, step-1] - consumption[component_num, step-1]<=0:
#                     stubble_availability[component_num, step]=0
#                 else: stubble_availability[component_num, step]=stubble_availability[component_num, step-1] - consumption[component_num, step-1]
#             #weighted availability (weight by consumption preference)
#             for component, proportion,component_num in zip(component_proportion.keys(),component_proportion.values(),range(len(component_proportion))):
#                 weighted_availability[component_num, step] = stubble_availability[component_num,step] * grazing_pref_component[component]
#             weighted_availability[5, step] = weighted_availability[:,step].sum()
#             #consumption per time step (consumption of each component %)
#             for component, proportion,component_num in zip(component_proportion.keys(),component_proportion.values(),range(len(component_proportion))):
#                 if weighted_availability[number_of_components,step] <= 0:
#                     consumption[component_num, step] = 0
#                 else:
#                     consumption[component_num, step] = (pinp.stubble['step_size']
#                     / weighted_availability[number_of_components,step] * weighted_availability[component_num, step] )
#             #cumulative comsumption
#             for component, proportion,component_num in zip(component_proportion.keys(),component_proportion.values(),range(len(component_proportion))):
#                 cumulative_consumption[component_num, step]= consumption[component_num].sum()
#
#         #determine the proportion of each component in each category
#         num_stub_cat = stub_cat_prop_ks1.shape[1]
#         categ_sizes = stub_cat_prop_ks1[crp,:]
#         cumulative_cat_size=[]
#         for i,j in zip(categ_sizes,range(num_stub_cat)):
#             if j > 0:
#                 cumulative_cat_size.append(cumulative_cat_size[-1]+i)
#             else: cumulative_cat_size.append(i)
#         #create numpy to store stubble dets that go into the rest of the stubble calcs
#         stub_cat_component_proportion = np.zeros([number_of_components,num_stub_cat])
#         for cat_num, cum_cat_size, cat_size in zip(range(num_stub_cat), cumulative_cat_size, categ_sizes):
#             for component in range(number_of_components):
#                 #ammount of a component consumed in a given category
#                 if cat_num == 0: #if not cat A then need to subtract off the consumed amount in the periods before
#                     comp_consumed = cumulative_consumption[component,round(cum_cat_size*100-1)]  #multiplied by 100 to convert the percent to int. it is then use to index the steps in the numpy arrays above, minus 1 because indexing starts from 0
#                 else: comp_consumed = (cumulative_consumption[component,round(cum_cat_size*100-1)] #use the cat list so that i can determine the the consumption of a component at in the cat before
#                     - cumulative_consumption[component,round(list(cumulative_cat_size)[cat_num-1]*100-1)])
#                 stub_cat_component_proportion[component, cat_num] = comp_consumed/cat_size/100
#         return stub_cat_component_proportion
#
#     def objective(x):
#         #multiplies the component dmd by the proportion of that component consumed in each cat
#         #this determines the overall dmd of that cat.
#         #the objective func minimised the diff between the value above and the inputted value of cat dmd
#         # component_dmd = np.array(component_dmd, dtype=float)
#         cat_a_component_propn=stubble_sim(x)[:,0]
#         a=np.dot(cat_a_component_propn,dmd_component_ks0[crp,:])
#         cat_b_component_propn=stubble_sim(x)[:,1]
#         b=np.dot(cat_b_component_propn,dmd_component_ks0[crp,:])
#         cat_c_component_propn=stubble_sim(x)[:,2]
#         c=np.dot(cat_c_component_propn,dmd_component_ks0[crp,:])
#         cat_d_component_propn=stubble_sim(x)[:,3]
#         d=np.dot(cat_d_component_propn,dmd_component_ks0[crp,:])
#         cat_a_target = pinp.stubble['stub_cat_qual'][crp,0]
#         cat_b_target = pinp.stubble['stub_cat_qual'][crp,1]
#         cat_c_target = pinp.stubble['stub_cat_qual'][crp,2]
#         cat_d_target = pinp.stubble['stub_cat_qual'][crp,3]
#
#         return ((a-cat_a_target)**2+(b-cat_b_target)**2+(c-cat_c_target)**2+(d-cat_d_target)**2)
#     #initial guesses
#     x0 = np.ones(7)
#     # bounds on variables
#     bndspositive = (0, 100.0) #qualtity of other components must be greater than 10%
#     no_upbnds = (1, 1.0e10) #pref has to be greater than stem
#     if index_k[crp] in ('r', 'z', 'l', 'f'):   #because these crops only have 4 stubble components ie no sheath
#         var_bound = (0,10) #still need to give optimisation some room to move otherwise it gives bad solution.
#     else: var_bound = (0,100)
#     bnds = (bndspositive, var_bound, bndspositive, no_upbnds, no_upbnds, no_upbnds, no_upbnds)
#     #may have to change around the solver (method) to get the best solution
#     solution = minimize(objective, x0, method='SLSQP', bounds=bnds)
#     x = solution.x
#     stub_cat_component_proportion = pd.DataFrame(stubble_sim(x))
#     stub_cat_component_proportion.to_excel(writer, sheet_name=index_k[crp],index=False,header=False)
#
#     #################################################
#     #post calcs to make sure everything looks good  #
#     #################################################
#     #check the component proportion
#     component_proportion={'grain' : grain_prop()
#         ,'blade' : x[0]
#         ,'sheath': x[1]
#         ,'chaff' : x[2]
#         ,'stem': 100- (x[0]+x[1]+x[2]+grain_prop())}
#     grazing_pref_component={'grain' :  x[3]
#         ,'blade' : x[4]
#         ,'sheath': x[5]
#         ,'chaff' : x[6]
#         ,'stem' :1}
#
#     def cat_ddm(x):
#         #multiplies the component dmd by the proportion of that component consumed in each cat
#         #this determines the overall dmd of that cat.
#         #the objective func minimised the diff between the value above and the inputted value of cat dmd
#         cat_a_dmd=stubble_sim(x)[:,0]
#         a=np.dot(cat_a_dmd,dmd_component_ks0[crp,:])
#         cat_b_dmd=stubble_sim(x)[:,1]
#         b=np.dot(cat_b_dmd,dmd_component_ks0[crp,:])
#         cat_c_dmd=stubble_sim(x)[:,2]
#         c=np.dot(cat_c_dmd,dmd_component_ks0[crp,:])
#         cat_d_dmd=stubble_sim(x)[:,3]
#         d=np.dot(cat_d_dmd,dmd_component_ks0[crp,:])
#         return(a,b,c,d)
#
#     print('-'*100)
#     print(index_k[crp])
#     print('component proportions at harv : ',component_proportion.values()) #dict values, check to make sure they look sensible
#     print('graxing pref : ',grazing_pref_component.values()) #dict values, check to make sure they look sensible
#     print('cat ddm : ',cat_ddm(x))
#     print('Target cat ddm : ',pinp.stubble['stub_cat_qual'][crp,0],pinp.stubble['stub_cat_qual'][crp,1], pinp.stubble['stub_cat_qual'][crp,2],pinp.stubble['stub_cat_qual'][crp,3])
#     print('objective : ',objective(x))
#
# writer.save()
#
#