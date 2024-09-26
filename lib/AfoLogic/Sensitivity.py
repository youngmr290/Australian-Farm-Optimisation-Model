# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 10:57:56 2020


@author: young

This is where all sensitivity values must be initialised.

"""

import numpy as np

from . import PropertyInputs as pinp
from . import UniversalInputs as uinp
from . import StructuralInputs as sinp
from . import Periods as per

##create dict - store sa variables in dict so they can easily be changed in the exp loop
sam = dict()
sap = dict()
saa = dict()
sat = dict()
sav = dict()
sar = dict()

def create_sa():
    '''
    Initialises default SA arrays. This gets done each loop in case property inputs changes and to clear last
    trial SA.'''
    ##len - mostly SA arrays can be initialised using the shape of the array they will be applied to.
    ## the length below need to be the full axis length before masking.
    len_b0 = np.count_nonzero(sinp.stock['i_mask_b0_b1'])
    len_b1 = len(sinp.stock['i_mask_b0_b1'])
    len_d = len(pinp.sheep['i_d_idx'])
    len_g0 = sinp.stock['i_mask_g0g3'].shape[0]
    len_g1 = sinp.stock['i_mask_g1g3'].shape[0]
    len_g2 = sinp.stock['i_mask_g2g3'].shape[0]
    len_g3 = sinp.stock['i_mask_g3g3'].shape[0]
    len_h2 = pinp.sheep['i_h2_len']
    len_h5 = pinp.sheep['i_h5_len']
    len_h7 = pinp.sheep['i_husb_operations_triggerlevels_h5h7h2'].shape[-1]
    len_i = pinp.sheep['i_i_len']
    len_k = len(sinp.general['i_idx_k'])
    len_crop_k = len(sinp.general['i_idx_k1'])
    len_pas_k = len(sinp.general['i_idx_k2'])
    len_crop_and_sup_k4 = len(uinp.price['grain_price_info'])
    len_k0 = pinp.sheep['i_k0_len'] #Weaning option
    len_k1 = pinp.sheep['i_k1_len'] #Oestrus cycle
    len_k2 = pinp.sheep['i_k2_len'] #LSLN cluster
    len_k3 = pinp.sheep['i_k3_len'] #dam age cluster
    len_k4 = pinp.sheep['i_k4_len'] #gender
    len_k5 = pinp.sheep['i_k5_len'] #BTRT cluster
    len_l = len(pinp.general['i_lmu_idx'])
    len_l0 = uinp.parameters['i_cl0_len2']
    len_n = len(uinp.general['i_fert_idx'])
    len_n1 = len(uinp.general['i_chem_idx'])
    len_o = pinp.sheep['i_o_len']
    len_R = 5000 #just use a big number - it is cut down later (this is because the length of r is not known because it can be affected by SA)
    len_s = pinp.sheep['i_s_len'] #s = shear
    len_s7 = len(uinp.sheep['i_salep_price_max_s7']) #s7 = sale grid
    len_t = len(pinp.general['pas_inc_t'])
    len_t1 = pinp.sheep['i_n_dam_sales'] + len_g0
    len_t2 = pinp.sheep['i_t2_len']
    len_T3 = 15 #this can get changed with sav so enter a big number and slice in the code.
    len_P = 500  #Capital P because it is an (over) estimate to initialise the p axes that will be sliced when len_p is known.
    len_p6 = len(pinp.period['i_fp_idx'])
    len_P7 = 10 #number of season node - use a big number because len_p7 can be adjusted by SA (if using MP model)
    len_Q = 20 #number of years in MP model - use a big number because len_q can be adjusted by SA
    len_V = 50  #Capital V because it is an (over) estimate to initialise the v axes that will be sliced when len_v is known.
    len_max_W1 = 3125 #number of nut options (i_nut_spread_n1 ** n_fvp) (this used to be calculated but the max possible was too big. This now assumes max n=5 and max fvp =5) #the max number of options for each starting w if all n and fvps included.
    len_max_W3 = 3125 #number of nut options (i_nut_spread_n3 ** n_fvp) (this used to be calculated but the max possible was too big. This now assumes max n=5 and max fvp =5) #the max number of options for each starting w if all n and fvps included.
    len_x = pinp.sheep['i_x_len']
    len_y = int(np.ceil(sinp.stock['i_age_max']))
    len_z = len(pinp.general['i_mask_z'])
    
    
    ###########
    # Global  #
    ###########
    
    ##Global
    sap['pi']=0 #global potential intake (this increases animal intake without altering animal energy profile, to alter the energy profile use ci[1:2,...]




    ##########
    #general #
    ##########
    ##SAV
    sav['steady_state']      = '-'                  #SA to alter if the model is steady state
    sav['mask_z']      = np.full_like(pinp.general['i_mask_z'], '-', dtype=object)   #SA to alter which seasons are included
    sav['prob_z']      = np.full_like(pinp.general['i_mask_z'], '-', dtype=object)   #SA to alter which seasons are included
    sav['date_node_p7']      = np.full(2, '-', dtype=object) #SA to alter p7 period dates (for MP model). Len 2 because p7 has two periods in the MP model. Note the node periods need to line up with p5 and p6 periods
    sav['inc_node_periods']      = '-'              #SA to alter if season nodes are included in the steady state model (note they are always included in the dsp version this only effects if they are included in steady state)
    sav['node_is_fvp'] = np.full(len_P7, '-', dtype=object) #SA to alter if season nodes are used as FVPs. This is generally True in the MP model.
    sav['seq_len']      = '-'                     #SA to alter the length of the season sequence in the SQ model
    sav['model_is_MP']      = '-'                 #SA to control when the MP framework is used.
    sav['MP_setup_trial_name']      = '-'         #SA to specify the name of the trial that generated the initial position for the MP run.
    sav['len_planning_horizon']      = '-'        #length of the planning horizon (only makes a difference if q is active eg in the MP model). This is used to weight q in the MP model.
    sav['inc_discount_factor']      = '-'         #SA to control if a discount factor (time value of money) is included. Default is false because not required for SE model but this should be set to True for MP model.
    sav['rev_update']      = '-'                  #SA to alter if the trial is being used to create rev std values
    sav['rev_number']      = '-'                  #SA to alter rev number - rev number is appended to the std rev value pkl file and can be used to select which rev is used as std for a given trial.
    sav['rev_trait_scenario'] = np.full_like(sinp.structuralsa['i_rev_trait_scenario'], '-', dtype=object) #SA value for which traits are to be held constant in REV analysis.
    sav['fs_create_pkl']      = '-'                  #SA to control if the trial is being used to create pkl fs
    sav['fs_create_number']      = '-'                  #SA to alter fs number - fs number is appended to the fs pkl file and can be used to select which pkl fs is created for a given trial.
    sav['gen_with_t']      = '-'                  #SA to control if sheep generator is run with active t axis.
    sav['fs_use_pkl']      = '-'                  #SA to control if the pkl fs is used or the excel input fs is used.
    sav['fs_use_number']      = '-'                  #SA to alter fs number - fs number is appended to the fs pkl file and can be used to select which pkl fs is used for a given trial.
    sav['r2adjust_inc']      = '-'              #SA to control if the r2 feedsupply adjustment from Excel is included.
    sav['inc_c1_variation'] = '-'               #control if price variation is on. This only effects result if risk aversion is included.
    sav['inc_risk_aversion'] = '-'              #control if risk aversion is included. Default is not included (ie utility=profit).
    sav['utility_method'] = '-'              #control which utility function is used
    sav['cara_risk_coef'] = '-'              #control risk coefficient for CRRA method
    sav['crra_risk_coef'] = '-'              #control risk coefficient for CRRA method
    sav['pinp_rot'] = '-'                       #control if using the pinp rotations or the full rotation list (note full rot requires simulation inputs)
    sav['crop_landuse_inc_k1'] = np.full(len(pinp.general['i_crop_landuse_inc_k1']), '-', dtype=object)    #control which crop landuses are included
    sav['pas_landuse_inc_k2'] = np.full(len(pinp.general['i_pas_landuse_inc_k2']), '-', dtype=object)     #control which pasture landuses are included
    sav['lmu_area_l']    = np.full(len(pinp.general['i_lmu_area']), '-', dtype=object)  # SA for area of each LMU
    sav['non_cropable_area_l']    = np.full(len(pinp.general['i_lmu_area']), '-', dtype=object)  # SA for area of each LMU that cant be cropped
    sav['lmu_arable_propn_l']    = np.full(len(pinp.general['i_lmu_area']), '-', dtype=object)  # SA for area of each LMU
    sav['phase_can_increase_kp7'] = np.full((len_k, len_P7), '-', dtype=object)  #SA to control when phases can be increased and reduced (only matters for dual season cropping)
    sav['phase_can_reduce_kp7'] = np.full((len_k, len_P7), '-', dtype=object)  #SA to control when phases can be decreased and reduced (only matters for dual season cropping)
    ##SAM
    sam['random'] = 1.0   # SA multiplier used to tweak any random variable when debugging or checking something (after being used it is best to revert the code)
    ##SAP
    ##SAA
    saa['random'] = 1.0   # SA addition used to tweak any random variable when debugging or checking something (after being used it is best to revert the code )
    ##SAT
    ##SAR

    ##########
    #finance #
    ##########
    ##SAV
    sav['working_cap_constraint_included'] = '-' #SA to control inclusion of work cap constraint in corepyomo
    sav['minroe']      = '-'                  #SA to alter the minroe (applied to both steady-state and dsp minroe inputs)
    sav['capital_limit']      = '-'          #SA to alter the capital limit (amount of money that can be loaned from bank)
    sav['interest_rate']      = '-'           #SA to alter the credit and debit interest from bank
    sav['opp_cost_capital']      = '-'        #SA to alter the opportunity cost of capital
    sav['fixed_dep_rate'] = '-'               #SA to alter the fixed rate of machinery depreciation per year
    sav['equip_insurance_rate'] = '-'         #SA to alter the insurance cost (% of machine value)
    sav['overheads'] = np.full(len(pinp.general['i_overheads']), '-', dtype=object)  #SA to alter the overhead costs
    ##SAM
    ##SAP
    ##SAA
    ##SAT
    ##SAR

    ########
    #Price #
    ########
    ##SAV
    sav['grain_percentile'] = '-'  #grain price percentile
    sav['grainp_k'] = np.full(len_crop_and_sup_k4, '-', dtype=object)   # SA value for grain prices for each crop for selected percentile (i.e. overwrites calculated price)
    sav['hayp_k'] = np.full(len_crop_and_sup_k4, '-', dtype=object)   # SA value for baled prices for each crop for selected percentile (i.e. overwrites calculated price)
    sav['woolp_mpg_percentile'] = '-'               #sa value for the wool price percentile
    sav['woolp_mpg'] = '-'                          # sa value for wool price at std micron for selected percentile (i.e. overwrites calculated price)
    sav['woolp_fdprem_percentile'] = '-'            # sa value for fd premium percentile (premium received by fd compared to std)
    sav['woolp_fdprem'] = '-'                       # sa value for fd premium
    sav['salep_percentile'] = '-'                   #Value for percentile for all sale grids
    sav['salep_max_s7'] = np.full(len_s7, '-', dtype=object)    #max sale price in grid for selected percentile (i.e. overwrites calculated price)
    sav['manager_cost'] = '-' #SA value for manager cost per year
    sav['permanent_cost'] = '-' #SA value for permanent cost per year
    sav['casual_cost'] = '-' #SA value for casual cost per hour
    sav['sale_ffcfw_min'] = np.full(len_s7, '-', dtype=object)        #min weight for sale in grid
    sav['sale_ffcfw_max'] = np.full(len_s7, '-', dtype=object)        #max weight for sale in grid
    ##SAM
    sam['grainp_k'] = np.ones(len_crop_and_sup_k4, dtype='float64')   # SA multiplier for grain prices for each crop
    sam['q_grain_price_scalar_Qk'] = np.ones((len_Q, len_crop_and_sup_k4), dtype='float64')   # SAM for grain price with q axis
    sam['q_wool_price_scalar_Q'] = np.ones(len_Q, dtype='float64')   # SAM for wool price with q axis
    sam['q_meat_price_scalar_Q'] = np.ones(len_Q, dtype='float64')   # SAM for meat price with q axis
    sam['woolp_mpg'] = 1.0                      # sa multiplier for wool price at std micron
    sam['salep_max_s7'] = np.ones(len_s7, dtype='float64')        #max sale price in grid
    sam['salep_month_adjust_s7s9p4'] = np.ones(uinp.sheep['i_salep_months_priceadj_s7s9p4'].shape, dtype='float64')      #monthly sale price
    ##SAP
    ##SAA
    ##SAT
    sat['salep_weight_scalar'] = 0.0 #Scalar for LW impact across grid 1
    sat['salep_score_scalar'] = 0.0  #Scalar for score impact across the grid
    ##SAR


    #########
    #Labour #
    #########
    ##SAV
    sav['manager_ub'] = '-'  #manager upper bound
    sav['manager_lo'] = '-'  #manager lower bound
    sav['perm_ub'] = '-'  #perm upper bound
    sav['perm_lo'] = '-'  #perm lower bound
    sav['casual_ub'] = '-'  #casual upper bound all year except seeding and harv
    sav['casual_lo'] = '-'  #casual lower bound all year except seeding and harv
    sav['seedharv_casual_ub'] = '-'  #casual upper bound at seeding and harv
    sav['seedharv_casual_lo'] = '-'  #casual lower bound at seeding and harv
    ##SAM
    ##SAP
    ##SAA
    ##SAT
    ##SAR

    #########
    #Mach #
    #########
    ##SAV
    sav['mach_option'] = '-'                    #control which machine compliment is used
    sav['daily_seed_hours'] = '-'               #number of hours seeder can run for each day.
    sav['seeding_eff'] = '-'               #propn of seeding time when the seeder is not moving i.e. due to refilling.
    sav['seeding_delays'] = '-'               #propn of the seeding period when seeding cannot occur due to bad weather
    sav['daily_harvest_hours'] = '-'               #number of hours harvester can run for each day.
    sav['harv_eff'] = '-'               #propn of seeding time when the harv is not moving (e.g prep/greaseing harvester, moving paddocks, testing grain moisture, etc)
    sav['harv_delays'] = '-'               #propn of the harv period when harv cannot occur due to bad weather
    sav['spray_eff'] = '-'               #propn of spraying time when sprayer is not working e.g. filling up.
    sav['variable_dep_hr_seeding'] = '-'               #variable depn of seeding gear per machine hour of seeding
    sav['variable_dep_hr_harv'] = '-'               #variable depn of harvest gear per machine hour of harvest
    sav['variable_dep_hr_spraying'] = '-'               #variable depn of sprayer gear per machine hour of spraying
    sav['variable_dep_hr_spreading'] = '-'               #variable depn of spreading gear per machine hour of spreading

    for option in uinp.mach:
        sav['clearing_value', option] = np.full(len(uinp.mach[option]['clearing_value']), '-', dtype=object) #clearing sale value of each item of machinery
        sav['number_seeders', option] = '-'                                 #number of seeders
        sav['seeding_rate_base', option] = '-'                                  #seeding speed of wheat on base LMU (km/hr)
        sav['number_harvesters', option] = '-'                              #number of harvesters
        sav['harvest_rate', option] = np.full(len_crop_k, '-', dtype=object) #harvesting rate of each crop (t/hr)
        sav['spraying_rate', option] = '-'                        #speed (km/hr)
        sav['spreader_cap', option] = '-'                                   #capacity (m3)
        sav['spreader_width', option] = np.full(len_n, '-', dtype=object)   #width for each fert type (m)
        sav['spreading_speed', option] = '-'                                #speed (km/hr)
        sav['spreading_eff', option] = '-'                                  #paddock efficiency of harvesting (accounts for overlap)
    ##SAM
    ##SAP
    ##SAA
    ##SAT
    ##SAR

    ###########
    #Sup feed #
    ###########
    ##SAV
    sav['max_sup_selectivity'] = '-'  #control the maximum propn of potential intake used by supplement when paddock feeding.
    sav['inc_sup_selectivity'] = '-'  #control inclusion of the sup selectivity bnd (maximum propn of potential intake used by supplement when paddock feeding).
    sav['confinement_feeding_cost_factor'] = '-'  #reduction factor for sup feeding cost when in confinement
    sav['confinement_feeding_labour_factor'] = '-'  #reduction factor for sup feeding labour when in confinement
    ##SAM
    ##SAP
    ##SAA
    ##SAT
    ##SAR

    ##############
    #Cropgrazing #
    ##############
    ##SAV
    sav['cropgrazing_inc'] = '-'  #control if crop grazing is allowed
    sav['bnd_crop_grazing_intensity'] = '-'  #control the amount of crop consumed per hectare of crop that can be grazed (i.e. doesnt include a crop are if the crop can't be grazed).
    sav['cropgraze_propn_area_grazed_kl'] = np.full((len_crop_k, len_l), '-', dtype=object)  #control proportion of crop area that can be grazed.
    sav['cropgraze_yield_penalty_k'] = np.full((len_crop_k), '-', dtype=object)  #Reduction in yield per kg of crop consumed (if grazed early in the growing season after the crop is established).
    ##SAM
    sam['cropgraze_yield_penalty'] = 1.0   # SA multiplier for the cropgraze yield penalty
    ##SAP
    ##SAA
    ##SAT
    ##SAR

    ####################
    #Salt land pasture #
    ####################
    ##SAV
    sav['slp_inc'] = '-'  #control if salt land pasture is included
    ##SAM
    sam['sb_growth'] = 1.0   # SA multiplier for the growth of saltbush on slp (applies to all lmus and fp)
    ##SAP
    ##SAA
    ##SAT
    ##SAR

    ####################
    #crop and rotation #
    ####################
    ##SAV
    sav['differentiate_wet_dry_seeding'] = '-'  #control is wet and dry seeding is differentiated - in the web app this is False meaning that all crops can be either dry or wet sown which removes the need to have special dry sown landuses.
    sav['user_rotphases'] = np.full(len_R, '-', dtype=object)  # SA value for the actual rotations - only used in web app - use capital R because rotation len from the web app can be different
    sav['rot_inc_R'] = np.full(len_R, '-', dtype=object)    # SA value for rotations included - web app - use capital R because rotation len from the web app can be different
    sav['sowing_freq_R'] = np.full(len_R, '-', dtype=object)    # SA value for pinp sowing frequency - use capital R because rotation len from the web app can be different
    sav['yield_Rz'] = np.full((len_R, len_z), '-', dtype=object)    # SA value for pinp grain/hay yield - use capital R because rotation len from the web app can be different
    sav['fert_R_nz'] = np.full((len_R, 4*len_z), '-', dtype=object)    # SA value for pinp fert - 4 because there are currently 4 ferts by r - use capital R because rotation len from the web app can be different
    sav['fert_passes_R_nz'] = np.full((len_R, 4*len_z), '-', dtype=object)    # SA value for pinp fert passses - 4 because there are currently 4 ferts by r - use capital R because rotation len from the web app can be different
    sav['chem_R_nz'] = np.full((len_R, 2*len_z), '-', dtype=object)    # SA value for pinp chem - 2 chem categorys in pinp (herb and fungicide). - use capital R because rotation len from the web app can be different
    sav['chem_passes_R_nz'] = np.full((len_R, len_n1*len_z), '-', dtype=object)    # SA value for pinp chem passes - use capital R because rotation len from the web app can be different
    sav['lmu_yield_adj_kl'] = np.full((len_crop_k, len_l), '-', dtype=object)    # SA value for yield adjustment by LMU
    sav['lmu_fert_adj_nl'] = np.full((len_n, len_l), '-', dtype=object)    # SA value for fert adjustment by LMU
    sav['lmu_chem_adj_l'] = np.full(len_l, '-', dtype=object)    # SA value for chem adjustment by LMU
    sav['lime_cost'] = '-'  #cost ($/ha) of lime
    sav['liming_freq'] = '-'  #number of years between applications
    ##SAM
    sam['q_crop_yield_scalar_Qk'] = np.ones((len_Q, len_crop_k), dtype='float64')  # SAM for grain price with q axis
    sam['crop_yield_k'] = np.ones(len_crop_k, dtype='float64')    # SA multiplier for all rotation yield
    sam['crop_fert_kn'] = np.ones((len_crop_k, len_n), dtype='float64') #SA multiplier on crop fertiliser
    sam['pas_fert_kn'] = np.ones((len_pas_k, len_n), dtype='float64') #SA multiplier on pas fertiliser
    sam['crop_chem_k'] = np.ones(len_crop_k, dtype='float64') #SA multiplier on crop chem package cost (ie all chem timing are scaled the same)
    sam['pas_chem_k'] = np.ones(len_pas_k, dtype='float64') #SA multiplier on pas chem package cost (ie all chem timing are scaled the same)
    sam['sowing_penalty'] = 1.0  #sam on sowing timeliness yield penalty
    ##SAP
    ##SAA
    saa['crop_fert_passes_kn'] = np.zeros((len_crop_k, len_n), dtype='float64') #SA adder on crop fertiliser passes
    saa['pas_fert_passes_kn'] = np.zeros((len_pas_k, len_n), dtype='float64') #SA adder on pas fertiliser passes
    saa['crop_chem_passes_kn1'] = np.zeros((len_crop_k, len_n1), dtype='float64') #SA adder on crop chem passes
    saa['pas_chem_passes_kn1'] = np.zeros((len_pas_k, len_n1), dtype='float64') #SA adder on pas chem passes
    ##SAT
    ##SAR

    ############
    # Pasture  #
    ############
    ##SAV
    sav['poc_inc'] = '-'  #control if poc is included
    sav['pas_inc_t'] = np.full_like(pinp.general['pas_inc_t'], '-', dtype=object) #SA value for pastures included mask
    ##SAM
    sam['q_pgr_scalar_Qp6'] = np.ones((len_Q, len_p6), dtype='float64')   # SAM for pgr with q axis
    sam['q_pgr_scalar'] = 1   # SAM for pgr from node[0] to the end of the GS (for mp model in the web app)

    for pasture in sinp.general['pastures'][pinp.general['i_pastures_exist']]:
        ##SAV
        ##SAM
        sam['germ',pasture]                    = 1.0                                                          # SA multiplier for germination on all lmus in all periods
        sam['germ_l',pasture]                  = np.ones((len(pinp.general['i_lmu_area'])),  dtype='float64')  # SA multiplier for germination on each lmus in all periods
        sam['pgr',pasture]                     = 1.0                                                          # SA multiplier for growth on all lmus in all periods
        sam['pgr_zp6',pasture]                   = np.ones((len_z, len_p6),  dtype='float64')  # SA multiplier for growth in each feed period
        sam['pgr_l',pasture]                   = np.ones((len(pinp.general['i_lmu_area'])),  dtype='float64')  # SA multiplier for growth on each lmus in all periods
        sam['dry_dmd_decline',pasture]         = 1.0                                                          # SA multiplier for the decline in digestibility of dry feed
        sam['grn_dmd_declinefoo_f',pasture]    = np.ones(len(pinp.period['i_fp_idx']),  dtype='float64')  # SA multiplier on decline in digestibility if green feed is not grazed (to increase FOO)
        sam['grn_dmd_range_f',pasture]         = np.ones(len(pinp.period['i_fp_idx']),  dtype='float64')  # SA multiplier on range in digestibility of green feed
        sam['grn_dmd_senesce_f',pasture]       = np.ones(len(pinp.period['i_fp_idx']),  dtype='float64')  # SA multiplier on reduction in digestibility when senescing
        sam['conservation_limit_f',pasture]    = np.ones(len(pinp.period['i_fp_idx']),  dtype='float64')  # SA multiplier for the conservation limit in each feed period
        ##SAP
        ##SAA pasture
        saa['germ',pasture]                    = 0.0                                                          # SA addition for germination on all lmus in all periods
        saa['pgr',pasture]                     = 0.0                                                          # SA addition for growth on all lmus in all periods
        saa['pgr_zp6',pasture]                   = np.zeros((len_z, len_p6),  dtype='float64')  # SA addition for growth in each feed period
        saa['pgr_l',pasture]                   = np.zeros((len(pinp.general['i_lmu_area'])),  dtype='float64')  # SA addition for growth on each lmus in all periods
        ##SAT
        ##SAR

    ############
    #livestock #
    ############
    ##SAV
    ###stock feedsupply
    sav['feedsupply_adj_r2p'] = np.full_like(pinp.feedsupply['i_feedsupply_adj_options_r2p'], '-', dtype=object)  # SA value for feedsupply adjustment.
    sav['dams_confinement_P'] = np.full(len_P, '-', dtype=object)  # SA to control the gen periods dams are in confimentment - this gets applied in FeedSupplyStock.py. Note, this will overwrite pkl so if using pkl to optimise confinement you most likely don’t want to use this SAV.
    sav['target_ebg_dams_Pb'] = np.full((len_P, len_b1), '-', dtype=object)  # SA to set lw target
    sav['target_ebg_offs_Pb'] = np.full((len_P, len_b0), '-', dtype=object)  # SA to set lw target
    ###stock others
    sav['nv_inc'] = '-'    #SA to store NV report values
    sav['lw_inc'] = '-'     #SA to store LW report values
    sav['ebw_inc'] = '-'  #SA to store EBW report values
    sav['wbe_inc'] = '-'  #SA to store EBW report values
    sav['cs_inc'] = '-'  #SA to store condition score report values
    sav['fs_inc'] = '-'  #SA to store fat score report values
    sav['onhand_mort_p_inc'] = '-'  #SA to store onhand report values
    sav['mort_inc'] = '-'  #SA to store mort report values
    sav['feedbud_inc'] = '-'  #SA to store feed budget report values
    sav['force_ebg_scalar'] = False  #SA to force scaling of energy components to ebg (active if components are the REV target traits)
    sav['age_max'] = '-'  #SA on length of the generator for dams (years)
    sav['age_max_offs'] = '-'    #SA on length of the generator for offspring (years)
    sav['eqn_compare']      = '-'                  #SA to alter if the different equation systems in the sheep sim are run and compared
    sav['eqn_used_g0_q1p7'] = np.full(uinp.sheep['i_eqn_used_g0_q1p7'].shape, '-', dtype=object) #SA value for which equation system to use
    sav['eqn_used_g1_q1p7'] = np.full(uinp.sheep['i_eqn_used_g1_q1p7'].shape, '-', dtype=object) #SA value for which equation system to use
    sav['eqn_used_g2_q1p7'] = np.full(uinp.sheep['i_eqn_used_g2_q1p7'].shape, '-', dtype=object) #SA value for which equation system to use
    sav['eqn_used_g3_q1p7'] = np.full(uinp.sheep['i_eqn_used_g3_q1p7'].shape, '-', dtype=object) #SA value for which equation system to use
    sav['TOL_inc']          = np.full(pinp.sheep['i_mask_i'].shape, '-', dtype=object)      # SA value for the inclusion of each TOL
    sav['date_shear_isxg0'] = np.full((len_i, len_s, len_x, len_g0), '-', dtype=object)      # SA value for the shearing sires
    sav['date_shear_isxg1'] = np.full((len_i, len_s, len_x, len_g1), '-', dtype=object)      # SA value for the shearing dams
    sav['date_shear_isxg3'] = np.full((len_i, len_s, len_x, len_g3), '-', dtype=object)      # SA value for the shearing offs
    sav['g3_included']      = np.full(pinp.sheep['i_g3_inc'].shape, '-', dtype=object)      # SA value for the inclusion of each offspring genotype
    sav['genotype']         = np.full(pinp.sheep['a_c2_c0'].shape, '-', dtype=object)       # this is the selection of the genotypes of the sires for B, M & T
    sav['scan_og1']         = np.full(pinp.sheep['i_scan_og1'].shape, '-', dtype=object)    # SA value for the scanning management option
    sav['nut_mask_dams_oWi'] = np.full((pinp.sheep['i_o_len'], len_max_W1, pinp.sheep['i_i_len']), '-', dtype=object)    #masks the nutrition options available e.g. high low high - the options selected are available for each starting weight (ie len_W = len_w/n_start_weights). This array is cut down in the code to the correct w len.
    sav['nut_mask_offs_sWix'] = np.full((pinp.sheep['i_s_len'], len_max_W3, pinp.sheep['i_i_len'], pinp.sheep['i_x_len']), '-', dtype=object)   #masks the nutrition options available e.g. high low high - the options selected are available for each starting weight (ie len_W = len_w/n_start_weights). This array is cut down in the code to the correct w len.
    sav['nut_spread_n1'] = np.full(sinp.structuralsa['i_nut_spread_n1'].shape, '-', dtype=object)      #nut spread dams
    sav['confinement_n1'] = np.full(sinp.structuralsa['i_confinement_n1'].shape, '-', dtype=object)    #bool array - This control allows confinement to occur if it is turned on for the given p6 period (controlled in feedsupply in property inputs)
    sav['nut_spread_n3'] = np.full(sinp.structuralsa['i_nut_spread_n3'].shape, '-', dtype=object)      #nut spread offs
    sav['confinement_n3'] = np.full(sinp.structuralsa['i_confinement_n3'].shape, '-', dtype=object)    #bool array - This control allows confinement to occur if it is turned on for the given p6 period (controlled in feedsupply in property inputs)
    sav['n_fs_dams'] = '-'      #nut options dams
    sav['n_fs_offs'] = '-'      #nut options offs
    sav['n_initial_lw_dams'] = '-'      #number of initial lws dams - note with the current code this can only be 2 or 3
    sav['n_initial_lw_offs'] = '-'      #number of initial lws offs - note with the current code this can only be 2 or 3
    sav['adjp_lw_initial_w1'] = np.full(sinp.structuralsa['i_adjp_lw_initial_w1'].shape, '-', dtype=object)      #initial lw adjustment dams
    sav['adjp_cfw_initial_w1'] = np.full(sinp.structuralsa['i_adjp_cfw_initial_w1'].shape, '-', dtype=object)    #initial cfw adjustment dams
    sav['adjp_fd_initial_w1'] = np.full(sinp.structuralsa['i_adjp_fd_initial_w1'].shape, '-', dtype=object)      #initial fd adjustment dams
    sav['adjp_fl_initial_w1'] = np.full(sinp.structuralsa['i_adjp_fl_initial_w1'].shape, '-', dtype=object)      #initial fl adjustment dams
    sav['condense_at_seasonstart'] = '-'  # SA to alter if condensing occurs at season start. Default is False except in the MP model when this can be set to True so that core fvps can be masked out and just just the season nodes for fvps.
    sav['user_fvp_date_dams_iu'] = np.full(sinp.structuralsa['i_dams_user_fvp_date_iu'].shape, '-', dtype=object)      #SA to control user fvp dates.
    sav['user_fvp_date_dams_yiu'] = np.full((len_y,)+sinp.structuralsa['i_dams_user_fvp_date_iu'].shape, '-', dtype=object)      #SA to control user fvp dates.
    sav['mask_fvp_dams'] = np.full(sinp.structuralsa['i_fvp_mask_dams'].shape, '-', dtype=object)      #SA to mask optional fvps.
    sav['fvp_is_dvp_dams'] = np.full(sinp.structuralsa['i_dvp_mask_f1'].shape, '-', dtype=object)      #SA to control if optional fvp is a dvp (note: fvps don't need to be dvps, the only benefit is if new information is available e.g. if animals uncluster, which allows differential management).
    sav['user_fvp_date_offs_iu'] = np.full(sinp.structuralsa['i_offs_user_fvp_date_iu'].shape, '-', dtype=object)      #SA to control user fvp dates.
    sav['user_fvp_date_offs_yiu'] = np.full((len_y,)+sinp.structuralsa['i_offs_user_fvp_date_iu'].shape, '-', dtype=object)      #SA to control user fvp dates.
    sav['mask_fvp_offs'] = np.full(sinp.structuralsa['i_fvp_mask_offs'].shape, '-', dtype=object)      #SA to mask optional fvps.
    sav['fvp_is_dvp_offs'] = np.full(sinp.structuralsa['i_fvp_mask_offs'].shape, '-', dtype=object)      #SA to control if optional fvp is a dvp (note: fvps don't need to be dvps, the only benefit is if new information is available e.g. if animals uncluster, which allows differential management).
    sav['r1_izg1'] = np.full(pinp.sheep['ia_r1_zig1'].shape, '-', dtype=object)   #SA to change the base feed option selected for dams
    sav['r1_izg3'] = np.full(pinp.sheep['ia_r1_zig3'].shape, '-', dtype=object)   #SA to change the base feed option selected for offspring
    sav['r2_ik0g1'] = np.full(pinp.sheep['ia_r2_ik0g1'].shape, '-', dtype=object)   #SA to change the selected feed adjustments selected for the k0 axis (wean age) for dams
    sav['r2_ik0g3'] = np.full(pinp.sheep['ia_r2_ik0g3'].shape, '-', dtype=object)   #SA to change the selected feed adjustments selected for the k0 axis (wean age) for offs
    sav['r2_isk2g1'] = np.full(pinp.sheep['ia_r2_isk2g1'].shape, '-', dtype=object)   #SA to change the selected feed adjustments selected for the k2 axis (LSLN) for dams
    sav['r2_ik5g3'] = np.full(pinp.sheep['ia_r2_ik5g3'].shape, '-', dtype=object)   #SA to change the selected feed adjustments selected for the k5 axis (BTRT) for offs
    sav['period_is_report_p'] = np.full(500, '-', dtype=object)  #SA to adjust the periods reported in ebw, wbe & fat '_cut' reports
    sav['LTW_loops_increment'] = '-'                  #SA to Increment the number of LTW loops carried out in the code. The base is 2 loops with 0 increment but if using pkl fs or ltw_adj is 0 then base is 0 loops.
    sav['offs_sale_opportunities'] = '-'              #offspring sale opportunities per dvp (if using method 1)
    ##SAM
    sam['kg_adult'] = 1.0                             #energy efficiency of adults (zf2==1)
    sam['mr_adult'] = 1.0                             #Maintenance requirement of adults (zf2==1)
    sam['pi_adult'] = 1.0                             #Potential intake of adults (zf2==1)
    sam['mr_pw'] = 1.0                                #Maintenance requirement post weaning
    sam['pi_pw'] = 1.0                                #Potential intake post weaning
    sam['kg_yatf'] = 1.0                              #energy efficiency of yatf
    sam['mr_yatf'] = 1.0                              #Maintenance requirement of yatf
    sam['pi_yatf'] = 1.0                              #Potential intake of yatf
    sam['LTW_dams'] = 1.0                       #adjust impact of life time wool fleece effects
    sam['LTW_offs'] = 1.0                       #adjust impact of life time wool fleece effects
    sam['pi_post_adult'] = 1.0                        #Post loop potential intake of adults (zf2==1)
    sam['pi_post_yatf'] = 1.0                        #Post loop potential intake of yatf
    sam['chill_index'] = 1.0                        #intermediate sam on chill index - impact on lamb survival.
    sam['heat_loss'] = 1.0                          #intermediate sam on heat loss - impact on energy requirements (set to 0 in REV analyses)
    sam['rr_og1'] = np.ones(pinp.sheep['i_scan_og1'].shape, dtype='float64')    # reproductive rate by age. Use shape that has og1
    sam['wean_redn_ol0g2'] = np.ones((len_o, len_l0, len_g2), dtype='float64')  #Adjust the number of yatf transferred at weaning - this is a high level sa, it impacts within a calculation not on an input
    ##SAP
    sap['evg'] = 0.0               #energy content of liveweight gain - this is a high level sa, it impacts within a calculation not on an input. It was only implemented on adults now all animals
    sap['mortalityb'] = 0.0        #Scale the calculated base mortality (for all animals) - this is a high level sa, it impacts within a calculation not on an input
    sap['kg_post_adult'] = 0.0           #Post loop energy efficiency of adults (zf2==1)
    sap['kg_post_yatf'] = 0.0           #Post loop energy efficiency of yatf
    sap['mr_post_adult'] = 0.0           #Post loop maintenance requirement of adults (zf2==1)
    sap['mr_post_yatf'] = 0.0           #Post loop maintenance requirement of yatf
    sap['mortalityp_ol0g2'] = np.zeros((len_o, len_l0, len_g2), dtype='float64')  #Scale the calculated progeny mortality at birth relative - this is a high level sa, it impacts within a calculation not on an input
    ##SAA
    saa['husb_cost_h2'] = np.zeros(uinp.sheep['i_husb_operations_contract_cost_h2'].shape, dtype='float64')  #SA value for contract cost of husbandry operations.
    saa['husb_labour_l2h2'] = np.zeros(uinp.sheep['i_husb_operations_labourreq_l2h2'].shape, dtype='float64')  #units of the job carried out per husbandry labour hour
    saa['r1_izg1'] = np.zeros(pinp.sheep['ia_r1_zig1'].shape, dtype=int)   #SA to change the base feed option selected for dams
    saa['r1_izg3'] = np.zeros(pinp.sheep['ia_r1_zig3'].shape, dtype=int)   #SA to change the base feed option selected for offspring
    saa['r2_isk2g1'] = np.zeros(pinp.sheep['ia_r2_isk2g1'].shape, dtype=int)   #SA to change the base feed option selected for dams
    saa['r2_ik5g3'] = np.zeros(pinp.sheep['ia_r2_ik5g3'].shape, dtype=int)   #SA to change the base feed option selected for offspring
    saa['date_born1st_iog'] = np.zeros(pinp.sheep['i_date_born1st_iog2'].shape, dtype=int)  #SA to adjust lambing date (used for ewe lambs).
    saa['feedsupply_r1jp'] = np.zeros(pinp.feedsupply['i_feedsupply_options_r1j2p'].shape, dtype='float64')  #SA value for feedsupply.
    saa['feedsupply_adj_r2p'] = np.zeros(pinp.feedsupply['i_feedsupply_adj_options_r2p'].shape, dtype='float64')  #SA value for feedsupply adjustment.
    saa['littersize_og1'] = np.zeros((len_o, len_g1), dtype='float64')   #sa to the litter size this changes the propn of singles/twins and trips whilst keeping propn empty the same.
    saa['conception_og1'] = np.zeros((len_o, len_g1), dtype='float64')   #sa to adjust the proportion of ewes that are empty whilst keepping litter size (number of lambs / pregnant ewes) the same
    saa['preg_increment_ol0g1'] = np.zeros((len_o, len_l0, len_g1), dtype='float64')   #sa to adjust the conception of an individual b1 slice at conception, so that the value of an extra lamb conceived of a given birth type can be calculated. a value of 1 would transfer all available animals into the target slice
    saa['mortalityx_ol0g1'] = np.zeros((len_o, len_l0, len_g1), dtype='float64')  #Adjust the progeny mortality due to exposure at birth relative - this is a high level sa, it impacts within a calculation not on an input
    saa['mortalitye_ol0g1'] = np.zeros((len_o, len_l0, len_g1), dtype='float64')  #Scale the calculated dam mortality at birth. 0.1 (10%) would increase the (perinatal) mortality of progeny at birth by 10 percentage points eg if mortality was 20% it would increase to 30%. - this is a high level sa, it impacts within a calculation not on an input
    saa['rr_age_og1'] = np.zeros(pinp.sheep['i_scan_og1'].shape, dtype='float64')    # reproductive rate by age. Use shape that has og1
    saa['wean_wt'] = 0.0         #weaning weight adjustment of yatf. Note: WWt changes without any change in MEI
    saa['mortalityb'] = 0.0      #Adjust the base mortality - this is a high level sa, it impacts within a calculation not on an input
    saa['feedsupply_adj_dams_ro'] = np.zeros((6, len_o), dtype='float64') #user fs adjuster - used in web app (simplified version of feedsupply_adj_r2p)
    saa['feedsupply_adj_offs_p10'] = np.zeros((3), dtype='float64') #user offs fs adjuster - used in web app (simplified version of feedsupply_adj_r2p)
    ##SAT
    ##SAR

    #####################
    ##stock parameters  #
    #####################
    ##SAV
    sav['srw_c2'] = np.full(uinp.parameters['i_srw_c2'].shape, '-', dtype=object)  #SA value for srw of each c2 genotype.
    sav['sfw_c2'] = np.full(uinp.parameters['i_sfw_c2'].shape, '-', dtype=object)  #std fleece weight genotype params
    sav['sfd_c2'] = np.full(uinp.parameters['i_sfd_c2'].shape, '-', dtype=object)  #std fibre diameter genotype params
    sav['ci_c1c2'] = np.full(uinp.parameters['i_ci_c2'].shape, '-', dtype=object)  #intake params for genotypes
    sav['cl_c1c2'] = np.full(uinp.parameters['i_cl_c2'].shape, '-', dtype=object)  #lactation params for genotypes.
    sav['cp_c1c2'] = np.full(uinp.parameters['i_cp_c2'].shape, '-', dtype=object)  #pregnancy params for genotypes.
    sav['cw_c1c2'] = np.full(uinp.parameters['i_cw_c2'].shape, '-', dtype=object)  #wool growth params for genotypes
    sav['cg_c1c2'] = np.full(uinp.parameters['i_cg_c2'].shape, '-', dtype=object)  #weight gain params for genotypes.
    sav['cd_c1c2'] = np.full(uinp.parameters['i_cd_c2'].shape, '-', dtype=object)  #mortality params for genotypes.
    sav['cl0_c1c2'] = np.full(uinp.parameters['i_cl0_c2'].shape, '-', dtype=object)  #litter size genotype params for genotypes.
    sav['cu2_c1c2'] = np.full(uinp.parameters['i_cu2_c2'].shape, '-', dtype=object)  #lamb survival params for genotypes.
    ##SAM
    sam['ci_c1c2'] = np.ones(uinp.parameters['i_ci_c2'].shape, dtype='float64')  #intake params for genotypes
    sam['cl_c1c2'] = np.ones(uinp.parameters['i_cl_c2'].shape, dtype='float64')  # lactation params for genotypes
    sam['cm_c1c2'] = np.ones(uinp.parameters['i_cm_c2'].shape, dtype='float64')  #intake params for genotypes
    sam['sfw_c2'] = np.ones(uinp.parameters['i_sfw_c2'].shape, dtype='float64')   #std fleece weight genotype params
    sam['muscle_target_c2'] = np.ones(uinp.parameters['i_muscle_target_c2'].shape, dtype='float64')   #std muscle mass target genotype params
    sam['rr'] = 1.0                        #scanning percentage (adjust the standard scanning % for f_conception_ltw and within function for f_conception_cs
    sam['husb_cost_h2'] = np.ones(uinp.sheep['i_husb_operations_contract_cost_h2'].shape, dtype='float64')  #SA value for contract cost of husbandry operations.
    sam['husb_mustering_h2'] = np.ones(uinp.sheep['i_husb_operations_muster_propn_h2'].shape, dtype='float64')  #SA value for mustering required for husbandry operations.
    sam['husb_labour_l2h2'] = np.ones(uinp.sheep['i_husb_operations_labourreq_l2h2'].shape, dtype='float64')  #units of the job carried out per husbandry labour hour
    ##SAP
    ##SAA
    saa['sfd_c2'] = 0.0                     #std fibre diameter genotype params
    saa['srw_c2'] = 0.0                     #std reference weight genotype params
    saa['cg_c1c2'] = np.zeros(uinp.parameters['i_cg_c2'].shape, dtype='float64')  #SA value for weight gain params.
    saa['ck_c1c2'] = np.zeros(uinp.parameters['i_ck_c2'].shape, dtype='float64')  #SA value for energy efficiency params.
    saa['cl0_c1c2'] = np.zeros(uinp.parameters['i_cl0_c2'].shape, dtype='float64')  #SA value for litter size genotype params.
    saa['scan_std_c2'] = 0.0                #std scanning percentage of a genotype. Controls the MU repro, initial propn of sing/twin/trip prog required to replace the dams, the lifetime productivity of the dams as affected by their BTRT..
    saa['nlb_c2'] = 0.0                #std scanning percentage of a genotype. Controls the MU repro, initial propn of sing/twin/trip prog required to replace the dams, the lifetime productivity of the dams as affected by their BTRT..
    saa['rr'] = 0.0                    #reproductive rate/scanning percentage (adjust the standard scanning % for f_conception_ltw and within function for f_conception_cs
    saa['ss'] = 0.0                    #staple strength (adjust SS in sgen end of period)

    ##SAT
    sat['cb0_c2'] = np.zeros(uinp.parameters['i_cb0_c2'].shape, dtype='float64')  #BTRT params for genotypes
    ##SAR

    #####################
    ##REV               #
    #####################
    ##Note the REV specific SA's get applied for the specified age stage (if you dont care about age stage you can use any SA with the REV)

    ##SAV
    sav['rev_update']      = '-'                  #SA to alter if the trial is being used to create rev std values
    sav['rev_number']      = '-'                  #SA to alter rev number - rev number is appended to the std rev value pkl file and can be used to select which rev is used as std for a given trial.
    sav['rev_trait_scenario'] = np.full_like(sinp.structuralsa['i_rev_trait_scenario'], '-', dtype=object) #SA value for which traits are to be held constant in REV analysis.
    sav['rev_age_stage']      = '-'                  #SA to set the age range that the rev sensitivities get applied.

    ##SAM
    sam['rev_cfw'] = 1.0   #std fleece weight genotype params
    sam['rev_pi_scalar'] = 1.0                      #Proportion to scale PI if MEI is scaled by REV adjustments
    ##SAP
    ##SAA
    saa['rev_fd'] = 0.0                     #std fibre diameter genotype params
    saa['rev_srw'] = 0.0                    #std reference weight genotype params
    saa['rev_evg'] = 0.0                    #SA value for weight gain params.
    saa['rev_ss'] = 0.0                     #staple strength (adjust SS in sgen end of period)
    saa['rev_cfat'] = 0.0                   #carcase fat (adjust GR depth at sale time)
    saa['rev_mortalityb'] = 0.0      #Adjust the base mortality - this is a high level sa, it impacts within a calculation not on an input
    saa['rev_mortalityx_ol0g1'] = np.zeros((len_o, len_l0, len_g1), dtype='float64')  #Adjust the progeny mortality due to exposure at birth relative - this is a high level sa, it impacts within a calculation not on an input
    saa['rev_littersize_og1'] = np.zeros((len_o, len_g1), dtype='float64')  # sa to the litter size this changes the propn of singles/twins and trips whilst keeping propn empty the same.
    saa['rev_conception_og1'] = np.zeros((len_o, len_g1), dtype='float64')  # sa to adjust the proportion of ewes that are empty whilst keepping litter size (number of lambs / pregnant ewes) the same

    ##SAT
    ##SAR

    ##########
    ##Bounds #
    ##########
    ##SAV
    sav['bnd_slp_area_l'] = np.full(len_l, '-', dtype=object)  #control the area of slp on each lmu
    sav['bnd_sb_consumption_p6'] = np.full(len(pinp.period['i_fp_idx']), '-', dtype=object)  #upper bnd on the amount of sb consumed
    sav['bnd_crop_area'] = np.full((len_Q, len_crop_k), '-', dtype=object)  #crop area for bound. if all values are '-' the bnd won't be used (there is not bnd_inc control for this one)
    sav['bnd_crop_area_percent'] = np.full((len_Q, len_crop_k), '-', dtype=object)  #crop area percent of farm area. if all values are '-' the bnd won't be used (there is not bnd_inc control for this one)
    sav['bnd_total_legume_area_percent'] = '-'  #Control the total percent of legume area on farm.
    sav['bnd_biomass_graze_k1'] = np.full(len_crop_k, '-', dtype=object)  #biomass graze area for bound. if all values are '-' the bnd won't be used (there is not bnd_inc control for this one)
    sav['bnd_total_pas_area_percent'] = np.full(len_Q, '-', dtype=object)  #Control the total percent of pasture area on farm.
    sav['bnd_pas_area_l'] = np.full(len_l, '-', dtype=object)  #pasture area by lmu for bound. if all values are '-' the bnd won't be used (there is not bnd_inc control for this one)
    sav['bnd_landuse_area_klz'] = np.full((len_crop_k, len_l, len_z), '-', dtype=object)  #landuse area by lmu and z. if all values are '-' the bnd won't be used
    sav['bnd_sup_per_dse'] = '-'   #SA to control the supplement per dse (kg/dse)
    sav['bnd_propn_dams_mated_og1'] = np.full((len_d,) + pinp.sheep['i_g3_inc'].shape, '-', dtype=object)   #proportion of dams mated
    sav['est_propn_dams_mated_og1'] = np.full((len_d,) + pinp.sheep['i_g3_inc'].shape, '-', dtype=object)   #estimated proportion of dams mated - used when bnd_propn is default "-"
    sav['propn_mated_w_inc'] = '-'   #Control if the constraint on proportion mated includes 'w' set
    sav['bnd_drys_sold_o'] = np.full(pinp.sheep['i_dry_sales_forced_o'].shape, '-', dtype=object)   #SA to force drys to be sold
    sav['bnd_drys_retained_o'] = np.full(pinp.sheep['i_dry_retained_forced_o'].shape, '-', dtype=object)   #SA to force drys to be retained
    sav['est_drys_retained_scan_o'] = np.full(pinp.sheep['i_drys_retained_scan_est_o'].shape, '-', dtype=object)   #Estimate of the propn of drys sold at scanning
    sav['est_drys_retained_birth_o'] = np.full(pinp.sheep['i_drys_retained_birth_est_o'].shape, '-', dtype=object)   #Estimate of the propn of drys sold at birth
    sav['bnd_sale_twice_dry_inc'] = '-'   #SA to include the bound which forces twice dry dams to be sold
    sav['bnd_twice_dry_propn'] = '-'   #SA to change twice dry dam proportion
    sav['bnd_total_dams'] = '-'   #control the total number of dams at prejoining
    sav['lobnd_across_startw'] = '-'   #control if dam and offs lower bound is across start w (if tru this means each start w must have numbers - this is used in fs optimisation).
    sav['bnd_lo_dam_inc'] = '-'   #control if dam lower bound is on.
    sav['bnd_lo_dams_tog1'] = np.full((len_t1,) + (len_d,) + (len_g1,), '-', dtype=object)   #min number of dams
    sav['bnd_lo_dams_tVg1'] = np.full((len_t1,) + (len_V,) + (len_g1,), '-', dtype=object)   #min number of dams
    sav['bnd_up_dams_K2tog1'] = np.full((20, len_t1, len_d, len_g1,), '-', dtype=object)   #max number of dams
    sav['bnd_up_dams_K2tVg1'] = np.full((20, len_t1, len_V, len_g1,), '-', dtype=object)   #max number of dams
    sav['bnd_total_dams_scanned'] = '-'   #total dams scanned (summed over all dvps) - this also controls if bound is on.
    sav['bnd_propn_dam5_retained'] = '-'   #propn of 5yo dams retained - this also controls if bound is on.
    sav['bnd_lo_off_inc'] = '-'   #control if off lower bound is on.
    sav['bnd_lo_offs_Tsdxg3'] = np.full((len_T3,) + (len_s,) + (len_d,) + (len_x,) + (len_g3,), '-', dtype=object)   #min number of offs
    sav['bnd_up_off_inc'] = '-'   #control if off upper bound is on.
    sav['bnd_up_offs_Tsdxg3'] = np.full((len_T3,) + (len_s,) + (len_d,) + (len_x,) + (len_g3,), '-', dtype=object)   #max number of offs
    sav['bnd_up_prog_tdxg2'] = np.full((len_t2,) + (len_d,) + (len_x,) + (len_g2,), '-', dtype=object)   #max number of offs
    sav['bnd_sr_Qt'] = np.full((len_Q, len_t), '-', dtype=object)   #SA to fix stocking rate
    sav['bnd_lw_change'] = '-'   #target difference in LW compared to the base w (nut 0). Used in MP model. A positive value means animals must be heavier than the base w slice at the end of the first node. A negitive value means the animals must be lighter. This bnd is only active in q[1].
    sav['bnd_min_sale_age_wether_g3'] = np.full(pinp.sheep['i_g3_inc'].shape, '-', dtype=object)   #SA to set min age wether can be sold
    sav['bnd_max_sale_age_wether_g3'] = np.full(pinp.sheep['i_g3_inc'].shape, '-', dtype=object)   #SA to set max age wether can be sold
    sav['bnd_min_sale_age_female_g1'] = np.full(pinp.sheep['i_g3_inc'].shape, '-', dtype=object)   #SA to set min age a dam can be sold - BBT offspring can be sold but BBT dams can't (because they are BB)
    sav['bnd_min_sale_age_female_dg3'] = np.full((len_d,) + (len_g3,), '-', dtype=object)   #SA to set min age a female can be sold - used to bound prog & offs
    sav['bnd_max_sale_age_female_g3'] = np.full(pinp.sheep['i_g3_inc'].shape, '-', dtype=object)   #SA to set max age wether can be sold
    sav['rot_lobound_rl'] = np.full((len_R,) + (len_l,), '-', dtype=object)
    ##SAM
    ##SAP
    ##SAA
    ##SAT
    ##SAR



