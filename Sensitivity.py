# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 10:57:56 2020


@author: young

This is where all sensitivity values must be initialised.

"""

import numpy as np
import PropertyInputs as pinp
import UniversalInputs as uinp
import StructuralInputs as sinp
import Periods as per

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
    len_d = len(pinp.sheep['i_d_idx'])
    len_g0 = sinp.stock['i_mask_g0g3'].shape[0]
    len_g1 = sinp.stock['i_mask_g1g3'].shape[0]
    len_g2 = sinp.stock['i_mask_g2g3'].shape[0]
    len_g3 = sinp.stock['i_mask_g3g3'].shape[0]
    len_h2 = pinp.sheep['i_h2_len']
    len_h5 = pinp.sheep['i_h5_len']
    len_h7 = pinp.sheep['i_husb_operations_triggerlevels_h5h7h2'].shape[-1]
    len_i = pinp.sheep['i_i_len']
    len_k = len(sinp.landuse['C'])
    len_k0 = pinp.sheep['i_k0_len'] #Weaning option
    len_k1 = pinp.sheep['i_k1_len'] #Oestrus cycle
    len_k2 = pinp.sheep['i_k2_len'] #LSLN cluster
    len_k3 = pinp.sheep['i_k3_len'] #dam age cluster
    len_k4 = pinp.sheep['i_k4_len'] #gender
    len_k5 = pinp.sheep['i_k5_len'] #BTRT cluster
    len_l = len(pinp.general['i_lmu_idx'])
    len_l0 = uinp.parameters['i_cl0_len2']
    len_o = pinp.sheep['i_o_len']
    len_R = 5000 #just use a big number - it is cut down later (this is because the length of r is not known because it can be affected by SA)
    len_s = pinp.sheep['i_s_len'] #s = shear
    len_t1 = pinp.sheep['i_n_dam_sales'] + len_g0
    len_t2 = pinp.sheep['i_t2_len']
    len_t3 = pinp.sheep['i_t3_len']
    len_P = 500  #Capital P because it is an (over) estimate to initialise the p axes that will be sliced when len_p is known.
    len_V = 50  #Capital V because it is an (over) estimate to initialise the v axes that will be sliced when len_v is known.
    len_x = pinp.sheep['i_x_len']
    len_y = int(np.ceil(sinp.stock['i_age_max']))
    len_z = len(pinp.general['i_mask_z'])
    
    
    ############
    #SAP Global#
    ############
    
    ##Global
    sap['pi']=0 #global potential intake (this increases animal intake without altering animal energy profile, to alter the energy profile use ci[1:2,...]
    
    ######
    #SAM #
    ######
    ##general
    sam['random'] = 1.0   # SA multiplier used to tweak any random variable when debugging or checking something (after being used it is best to remove it)
    sam['grainp'] = 1.0   # SA multiplier for all grain prices
    sam['grainp_k'] = np.ones(len_k, dtype=np.float64)   # SA multiplier for grain prices for each crop
    
    ##crop
    sam['all_rot_yield'] = 1.0   # SA multiplier for all rotation yield
    
    ##saltbush
    sam['sb_growth'] = 1.0   # SA multiplier for the growth of saltbush on slp (applies to all lmus and fp)
    
    ## Annual module sensitivity variables - these need to have the same name for each pasture type
    sam['germ','annual']                    = 1.0                                                          # SA multiplier for germination on all lmus in all periods
    sam['germ','understory']                    = 1.0                                                          # SA multiplier for germination on all lmus in all periods
    sam['germ_l','annual']                  = np.ones((len(pinp.general['i_lmu_area'])),  dtype=np.float64)  # SA multiplier for germination on each lmus in all periods
    sam['germ_l','understory']                  = np.ones((len(pinp.general['i_lmu_area'])),  dtype=np.float64)  # SA multiplier for germination on each lmus in all periods
    sam['pgr','annual']                     = 1.0                                                          # SA multiplier for growth on all lmus in all periods
    sam['pgr','understory']                     = 1.0                                                          # SA multiplier for growth on all lmus in all periods
    sam['pgr_f','annual']                   = np.ones(len(pinp.period['i_fp_idx']),  dtype=np.float64)  # SA multiplier for growth in each feed period
    sam['pgr_f','understory']                   = np.ones(len(pinp.period['i_fp_idx']),  dtype=np.float64)  # SA multiplier for growth in each feed period
    sam['pgr_l','annual']                   = np.ones((len(pinp.general['i_lmu_area'])),  dtype=np.float64)  # SA multiplier for growth on each lmus in all periods
    sam['pgr_l','understory']                   = np.ones((len(pinp.general['i_lmu_area'])),  dtype=np.float64)  # SA multiplier for growth on each lmus in all periods
    sam['dry_dmd_decline','annual']         = 1.0                                                          # SA multiplier for the decline in digestibility of dry feed
    sam['dry_dmd_decline','understory']         = 1.0                                                          # SA multiplier for the decline in digestibility of dry feed
    sam['grn_dmd_declinefoo_f','annual']    = np.ones(len(pinp.period['i_fp_idx']),  dtype=np.float64)  # SA multiplier on decline in digestibility if green feed is not grazed (to increase FOO)
    sam['grn_dmd_declinefoo_f','understory']    = np.ones(len(pinp.period['i_fp_idx']),  dtype=np.float64)  # SA multiplier on decline in digestibility if green feed is not grazed (to increase FOO)
    sam['grn_dmd_range_f','annual']         = np.ones(len(pinp.period['i_fp_idx']),  dtype=np.float64)  # SA multiplier on range in digestibility of green feed
    sam['grn_dmd_range_f','understory']         = np.ones(len(pinp.period['i_fp_idx']),  dtype=np.float64)  # SA multiplier on range in digestibility of green feed
    sam['grn_dmd_senesce_f','annual']       = np.ones(len(pinp.period['i_fp_idx']),  dtype=np.float64)  # SA multiplier on reduction in digestibility when senescing
    sam['grn_dmd_senesce_f','understory']       = np.ones(len(pinp.period['i_fp_idx']),  dtype=np.float64)  # SA multiplier on reduction in digestibility when senescing
    sam['conservation_limit_f','annual']    = np.ones(len(pinp.period['i_fp_idx']),  dtype=np.float64)  # SA multiplier for the conservation limit in each feed period
    sam['conservation_limit_f','understory']    = np.ones(len(pinp.period['i_fp_idx']),  dtype=np.float64)  # SA multiplier for the conservation limit in each feed period
    
    ##livestock
    sam['woolp_mpg'] = 1.0                      # sa multiplier for wool price at std micron
    sam['salep_max'] = 1.0                      #max sale price in grid
    sam['salep_month_adjust_s7s9p4'] = np.ones(uinp.sheep['i_salep_months_priceadj_s7s9p4'].shape, dtype=np.float64)      #monthly sale price
    sam['kg'] = 1.0                             #energy efficiency of adults (zf2==1)
    sam['mr'] = 1.0                             #Maintenance requirement of adults (zf2==1)
    sam['pi'] = 1.0                             #Potential intake of adults (zf2==1)
    sam['LTW_dams'] = 1.0                       #adjust impact of life time wool fleece effects
    sam['LTW_offs'] = 1.0                       #adjust impact of life time wool fleece effects
    sam['pi_post'] = 1.0                        #Post loop potential intake of adults (zf2==1)
    sam['chill'] = 1.0                        #intermediate sam on chill.
    
    ##stock parameters
    sam['ci_c2'] = np.ones(uinp.parameters['i_ci_c2'].shape, dtype=np.float64)  #intake params for genotypes
    sam['sfw_c2'] = 1.0                         #std fleece weight genotype params
    sam['rr'] = 1.0                        #scanning percentage (adjust the standard scanning % for f_conception_ltw and within function for f_conception_cs
    sam['husb_cost_h2'] = np.ones(uinp.sheep['i_husb_operations_contract_cost_h2'].shape, dtype=np.float64)  #SA value for contract cost of husbandry operations.
    sam['husb_mustering_h2'] = np.ones(uinp.sheep['i_husb_operations_muster_propn_h2'].shape, dtype=np.float64)  #SA value for mustering required for husbandry operations.
    sam['husb_labour_l2h2'] = np.ones(uinp.sheep['i_husb_operations_labourreq_l2h2'].shape, dtype=np.float64)  #units of the job carried out per husbandry labour hour
    
    ######
    #SAP #
    ######
    sap['evg'] = 0.0               #energy content of liveweight gain - this is a high level sa, it impacts within a calculation not on an input and is only implemented on adults
    sap['mortalityp'] = 0.0        #Scale the calculated progeny mortality at birth relative - this is a high level sa, it impacts within a calculation not on an input
    sap['mortalitye'] = 0.0        #Scale the calculated dam mortality at birth - this is a high level sa, it impacts within a calculation not on an input
    sap['mortalityb'] = 0.0        #Scale the calculated base mortality - this is a high level sa, it impacts within a calculation not on an input
    sap['kg_post'] = 0.0           #Post loop energy efficiency of adults (zf2==1)
    sap['mr_post'] = 0.0           #Post loop maintenance requirement of adults (zf2==1)
    
    
    ######
    #SAA #
    ######
    ##pasture
    saa['germ','annual']                    = 0.0                                                          # SA addition for germination on all lmus in all periods
    saa['germ','understory']                    = 0.0                                                          # SA addition for germination on all lmus in all periods
    saa['pgr','annual']                     = 0.0                                                          # SA addition for growth on all lmus in all periods
    saa['pgr','understory']                     = 0.0                                                          # SA addition for growth on all lmus in all periods
    saa['pgr_f','annual']                   = np.zeros(len(pinp.period['i_fp_idx']),  dtype=np.float64)  # SA addition for growth in each feed period
    saa['pgr_f','understory']                   = np.zeros(len(pinp.period['i_fp_idx']),  dtype=np.float64)  # SA addition for growth in each feed period
    saa['pgr_l','annual']                   = np.zeros((len(pinp.general['i_lmu_area'])),  dtype=np.float64)  # SA addition for growth on each lmus in all periods
    saa['pgr_l','understory']                   = np.zeros((len(pinp.general['i_lmu_area'])),  dtype=np.float64)  # SA addition for growth on each lmus in all periods
    
    ##stock
    saa['husb_cost_h2'] = np.zeros(uinp.sheep['i_husb_operations_contract_cost_h2'].shape, dtype=np.float64)  #SA value for contract cost of husbandry operations.
    saa['husb_labour_l2h2'] = np.zeros(uinp.sheep['i_husb_operations_labourreq_l2h2'].shape, dtype=np.float64)  #units of the job carried out per husbandry labour hour
    saa['r1_izg1'] = np.zeros(pinp.sheep['ia_r1_zig1'].shape, dtype=int)   #SA to change the base feed option selected for dams
    saa['r1_izg3'] = np.zeros(pinp.sheep['ia_r1_zig3'].shape, dtype=int)   #SA to change the base feed option selected for offspring
    saa['r2_isk2g1'] = np.zeros(pinp.sheep['ia_r2_isk2g1'].shape, dtype=int)   #SA to change the base feed option selected for dams
    saa['r2_ik5g3'] = np.zeros(pinp.sheep['ia_r2_ik5g3'].shape, dtype=int)   #SA to change the base feed option selected for offspring
    saa['date_born1st_iog'] = np.zeros(pinp.sheep['i_date_born1st_iog2'].shape, dtype=int)  #SA to adjust lambing date (used for ewe lambs).
    saa['feedsupply_r1jp'] = np.zeros(pinp.feedsupply['i_feedsupply_options_r1j2p'].shape, dtype=np.float64)  #SA value for feedsupply.
    saa['feedsupply_adj_r2p'] = np.zeros(pinp.feedsupply['i_feedsupply_adj_options_r2p'].shape, dtype=np.float64)  #SA value for feedsupply adjustment.
    
    ##stock parameters
    saa['sfd_c2'] = 0.0                     #std fibre diameter genotype params
    saa['cl0_c2'] = np.zeros(uinp.parameters['i_cl0_c2'].shape, dtype=np.float64)  #SA value for litter size genotype params.
    saa['scan_std_c2'] = 0.0                #std scanning percentage of a genotype. Controls the MU repro, initial propn of sing/twin/trip prog required to replace the dams, the lifetime productivity of the dams as affected by their BTRT..
    saa['nlb_c2'] = 0.0                #std scanning percentage of a genotype. Controls the MU repro, initial propn of sing/twin/trip prog required to replace the dams, the lifetime productivity of the dams as affected by their BTRT..
    saa['rr'] = 0.0                    #reproductive rate/scanning percentage (adjust the standard scanning % for f_conception_ltw and within function for f_conception_cs
    saa['rr_age_og1'] = np.zeros(pinp.sheep['i_scan_og1'].shape, dtype=np.float64)    # reproductive rate by age. Use shape that has og1
    saa['mortalityx_ol0g1'] = np.zeros((len_o, len_l0, len_g1), dtype=np.float64)  #Adjust the progeny mortality due to exposure at birth relative - this is a high level sa, it impacts within a calculation not on an input
    saa['wean_wt'] = 0.0            #weaning weight adjustment of yatf. Note: WWt changes without any change in MEI
    
    ######
    #SAT #
    ######
    sat['salep_weight_scalar'] = 0.0 #Scalar for LW impact across grid 1
    sat['salep_score_scalar'] = 0.0  #Scalar for score impact across the grid
    
    
    ######
    #SAR #
    ######
    
    
    ######
    #SAV #
    ######
    ##if you initialise an array it must be type object (so that you can assign int/float/bool into the array)

    ##general
    sav['steady_state']      = '-'                  #SA to alter if the model is steady state
    sav['mask_z']      = np.full_like(pinp.general['i_mask_z'], '-', dtype=object)   #SA to alter which seasons are included
    sav['inc_node_periods']      = '-'              #SA to alter if season nodes are included in the steady state model (note they are always included in the dsp version this only effects if they are included in steady state)
    sav['seq_len']      = '-'                     #SA to alter the length of the season sequence in the SQ model
    sav['rev_create']      = '-'                  #SA to alter if the trial is being used to create rev std values
    sav['rev_number']      = '-'                  #SA to alter rev number - rev number is appended to the std rev value pkl file and can be used to select which rev is used as std for a given trial.
    sav['rev_trait_inc'] = np.full_like(sinp.structuralsa['i_rev_trait_inc'], '-', dtype=object) #SA value for which traits are to be held constant in REV analysis.
    sav['fs_create_pkl']      = '-'                  #SA to control if the trial is being used to create pkl fs
    sav['fs_create_number']      = '-'                  #SA to alter fs number - fs number is appended to the fs pkl file and can be used to select which pkl fs is created for a given trial.
    sav['gen_with_t']      = '-'                  #SA to control if sheep generator is run with active t axis.
    sav['fs_use_pkl']      = '-'                  #SA to control if the pkl fs is used or the excel input fs is used.
    sav['fs_use_number']      = '-'                  #SA to alter fs number - fs number is appended to the fs pkl file and can be used to select which pkl fs is used for a given trial.
    sav['use_pkl_condensed_start_condition'] = '-'  #SA to control if the pkl values are used for the start animal at condensing
    sav['r2adjust_inc']      = '-'              #SA to control if the r2 feedsupply adjustment from Excel is included.
    sav['inc_c1_variation'] = '-'               #control if price variation is on. This only effects result if risk aversion is included.
    sav['inc_risk_aversion'] = '-'              #control if risk aversion is included. Default is not included (ie utility=profit).
    sav['utility_method'] = '-'              #control which utility function is used
    sav['cara_risk_coef'] = '-'              #control risk coefficient for CRRA method
    sav['crra_risk_coef'] = '-'              #control risk coefficient for CRRA method
    sav['pinp_rot'] = '-'                       #control if using the pinp rotations or the full rotation list (note full rot requires simulation inputs)
    sav['mach_option'] = '-'                    #control which machine compliment is used
    sav['lmu_area_l']    = np.full(len(pinp.general['i_lmu_area']), '-', dtype=object)  # SA for area of each LMU

    ##finance
    sav['minroe']      = '-'                  #SA to alter the minroe (applied to both steady-state and dsp minroe inputs)
    sav['overdraw_limit']      = '-'          #SA to alter the overdraw limit (amount of money that can be loaned from bank)
    sav['interest_rate']      = '-'           #SA to alter the credit and debit interest from bank
    sav['opp_cost_capital']      = '-'        #SA to alter the opportunity cost of capital

    ##price
    sav['grain_percentile'] = '-'  #grain price percentile

    ##labour
    sav['casual_ub'] = '-'  #casual upper bound all year except seeding and harv
    sav['seedharv_casual_ub'] = '-'  #casual upper bound at seeding and harv

    ##sup feed
    sav['max_sup_selectivity'] = '-'  #control the maximum propn of potential intake used by supplement when paddock feeding.

    ##cropgrazing
    sav['cropgrazing_inc'] = '-'  #control if crop grazing is allowed

    ##salt land pasture
    sav['slp_inc'] = '-'  #control if salt land pasture is included

    ##bounds
    sav['bnd_slp_area_l'] = np.full(len_l, '-', dtype=object)  #control the area of slp on each lmu
    sav['bnd_sb_consumption_p6'] = np.full(len(pinp.period['i_fp_idx']), '-', dtype=object)  #upper bnd on the amount of sb consumed
    sav['bnd_total_pas_area'] = '-'  #Total pasture area for bound. '-' is default so it will chuck an error if the bound is turned on without a specified area
    sav['bnd_pasarea_inc'] = '-'   #SA to turn on the pasture area bound
    sav['bnd_rotn_inc'] = '-'   #SA to turn on the phase area bounds
    sav['bnd_sr_inc'] = '-'   #SA to turn on the stocking rate bounds
    sav['bnd_sup_per_dse'] = '-'   #SA to control the supplement per dse (kg/dse)
    sav['bnd_propn_dams_mated_og1'] = np.full((len_d,) + pinp.sheep['i_g3_inc'].shape, '-', dtype=object)   #proportion of dams mated
    sav['est_propn_dams_mated_og1'] = np.full((len_d,) + pinp.sheep['i_g3_inc'].shape, '-', dtype=object)   #estimated proportion of dams mated - used when bnd_propn is default "-"
    sav['bnd_drys_sold_o'] = np.full(pinp.sheep['i_dry_sales_forced_o'].shape, '-', dtype=object)   #SA to force drys to be sold
    sav['bnd_drys_retained_o'] = np.full(pinp.sheep['i_dry_retained_forced_o'].shape, '-', dtype=object)   #SA to force drys to be retained
    sav['est_drys_retained_scan_o'] = np.full(pinp.sheep['i_drys_retained_scan_est_o'].shape, '-', dtype=object)   #Estimate of the propn of drys sold at scanning
    sav['est_drys_retained_birth_o'] = np.full(pinp.sheep['i_drys_retained_birth_est_o'].shape, '-', dtype=object)   #Estimate of the propn of drys sold at birth
    sav['bnd_sale_twice_dry_inc'] = '-'   #SA to include the bound which forces twice dry dams to be sold
    sav['bnd_twice_dry_propn'] = '-'   #SA to change twice dry dam proportion
    sav['bnd_lo_dam_inc'] = '-'   #control if dam lower bound is on.
    sav['bnd_lo_dams_tog1'] = np.full((len_t1,) + (len_d,) + (len_g1,), '-', dtype=object)   #min number of dams
    sav['bnd_lo_dams_tVg1'] = np.full((len_t1,) + (len_V,) + (len_g1,), '-', dtype=object)   #min number of dams
    sav['bnd_up_dam_inc'] = '-'   #control if dam upper bound is on.
    sav['bnd_up_dams_tog1'] = np.full((len_t1,) + (len_d,) + (len_g1,), '-', dtype=object)   #max number of dams
    sav['bnd_up_dams_tVg1'] = np.full((len_t1,) + (len_V,) + (len_g1,), '-', dtype=object)   #max number of dams
    sav['bnd_total_dams_scanned'] = '-'   #total dams scanned (summed over all dvps) - this also controls if bound is on.
    sav['bnd_propn_dam5_retained'] = '-'   #propn of 5yo dams retained - this also controls if bound is on.
    sav['bnd_lo_off_inc'] = '-'   #control if off lower bound is on.
    sav['bnd_lo_offs_tsdxg3'] = np.full((len_t3,) + (len_s,) + (len_d,) + (len_x,) + (len_g3,), '-', dtype=object)   #min number of offs
    sav['bnd_up_off_inc'] = '-'   #control if off upper bound is on.
    sav['bnd_up_offs_tsdxg3'] = np.full((len_t3,) + (len_s,) + (len_d,) + (len_x,) + (len_g3,), '-', dtype=object)   #max number of offs
    sav['bnd_up_prog_inc'] = '-'   #control if prog upper bound is on.
    sav['bnd_up_prog_tdxg2'] = np.full((len_t2,) + (len_d,) + (len_x,) + (len_g2,), '-', dtype=object)   #max number of offs
    sav['bnd_sr_t'] = np.full(pinp.sheep['i_sr_constraint_t'].shape, '-', dtype=object)   #SA to fix stocking rate
    sav['bnd_min_sale_age_wether_g3'] = np.full(pinp.sheep['i_g3_inc'].shape, '-', dtype=object)   #SA to set min age wether can be sold
    sav['bnd_max_sale_age_wether_g3'] = np.full(pinp.sheep['i_g3_inc'].shape, '-', dtype=object)   #SA to set max age wether can be sold
    sav['bnd_min_sale_age_female_g1'] = np.full(pinp.sheep['i_g3_inc'].shape, '-', dtype=object)   #SA to set min age a dam can be sold - BBT offspring can be sold but BBT dams can't (because they are BB)
    sav['bnd_min_sale_age_female_dg3'] = np.full((len_d,) + (len_g3,), '-', dtype=object)   #SA to set min age a female can be sold - used to bound prog & offs
    sav['bnd_max_sale_age_female_g3'] = np.full(pinp.sheep['i_g3_inc'].shape, '-', dtype=object)   #SA to set max age wether can be sold
    sav['rot_lobound_rl'] = np.full((len_R,) + (len_l,), '-', dtype=object)
    
    ##pasture
    sav['pas_inc_t'] = np.full_like(pinp.general['pas_inc'], '-', dtype=object) #SA value for pastures included mask
    
    ##Stock
    ###feedsupply
    sav['feedsupply_adj_r2p'] = np.full_like(pinp.feedsupply['i_feedsupply_adj_options_r2p'], '-', dtype=object)  # SA value for feedsupply adjustment.
    sav['dams_confinement_P'] = np.full(len_P, '-', dtype=object)  # SA to control the gen periods dams are in confimentment - this gets applied in FeedSupplyStock.py. Note, this will overwrite pkl so if using pkl to optimise confinement you most likely don’t want to use this SAV.
    ###others
    sav['nv_inc'] = '-'    #SA to store NV report values
    sav['lw_inc'] = '-'     #SA to store LW report values
    sav['ffcfw_inc'] = '-'  #SA to store FFCFW report values
    sav['onhand_mort_p_inc'] = '-'  #SA to store onhand report values
    sav['mort_inc'] = '-'  #SA to store mort report values
    sav['feedbud_inc'] = '-'  #SA to store feed budget report values
    sav['eqn_compare']      = '-'                  #SA to alter if the different equation systems in the sheep sim are run and compared
    sav['eqn_used_g0_q1p7'] = np.full(uinp.sheep['i_eqn_used_g0_q1p7'].shape, '-', dtype=object) #SA value for which equation system to use
    sav['eqn_used_g1_q1p7'] = np.full(uinp.sheep['i_eqn_used_g1_q1p7'].shape, '-', dtype=object) #SA value for which equation system to use
    sav['eqn_used_g2_q1p7'] = np.full(uinp.sheep['i_eqn_used_g2_q1p7'].shape, '-', dtype=object) #SA value for which equation system to use
    sav['eqn_used_g3_q1p7'] = np.full(uinp.sheep['i_eqn_used_g3_q1p7'].shape, '-', dtype=object) #SA value for which equation system to use
    sav['TOL_inc']          = np.full(pinp.sheep['i_mask_i'].shape, '-', dtype=object)      # SA value for the inclusion of each TOL
    sav['g3_included']      = np.full(pinp.sheep['i_g3_inc'].shape, '-', dtype=object)      # SA value for the inclusion of each offspring genotype
    sav['genotype']         = np.full(pinp.sheep['a_c2_c0'].shape, '-', dtype=object)       # this is the selection of the genotypes of the sires for B, M & T
    sav['scan_og1']         = np.full(pinp.sheep['i_scan_og1'].shape, '-', dtype=object)    # SA value for the scanning management option
    sav['woolp_mpg_percentile'] = '-'               #sa value for the wool price percentile
    sav['woolp_mpg'] = '-'                          # sa value for wool price at std micron
    sav['woolp_fdprem_percentile'] = '-'            # sa value for fd premium percentile (premium received by fd compared to std)
    sav['woolp_fdprem'] = '-'                       # sa value for fd premium
    sav['salep_percentile'] = '-'                   #Value for percentile for all sale grids
    sav['salep_max'] = '-'                          #max sale price in grid
    len_max_w1 = sinp.structuralsa['i_w_start_len1'] * len(sinp.structuralsa['i_nut_spread_n1']) ** (
            len(sinp.stock['i_fixed_fvp_mask_dams'])+len(sinp.structuralsa['i_fvp_mask_dams'])) #the max size of w if all n and fvps included.
    len_max_w3 = sinp.structuralsa['i_w_start_len3'] * len(sinp.structuralsa['i_nut_spread_n3']) ** len(sinp.structuralsa['i_fvp_mask_offs']) #the max size of w if all n and fvps included.
    sav['nut_mask_dams_oWi'] = np.full((pinp.sheep['i_o_len'], len_max_w1, pinp.sheep['i_i_len']), '-', dtype=object)    #masks the nutrition options available e.g. high low high - the options selected are available for each starting weight (ie len_W = len_w/n_start_weights). This array is cut down in the code to the correct w len.
    sav['nut_mask_offs_sWix'] = np.full((pinp.sheep['i_s_len'], len_max_w3, pinp.sheep['i_i_len'], pinp.sheep['i_x_len']), '-', dtype=object)   #masks the nutrition options available e.g. high low high - the options selected are available for each starting weight (ie len_W = len_w/n_start_weights). This array is cut down in the code to the correct w len.
    sav['nut_spread_n1'] = np.full(sinp.structuralsa['i_nut_spread_n1'].shape, '-', dtype=object)      #nut spread dams
    sav['confinement_n1'] = np.full(sinp.structuralsa['i_confinement_n1'].shape, '-', dtype=object)    #bool array - This control allows confinement to occur if it is turned on for the given p6 period (controlled in feedsupply in property inputs)
    sav['nut_spread_n3'] = np.full(sinp.structuralsa['i_nut_spread_n3'].shape, '-', dtype=object)      #nut spread offs
    sav['confinement_n3'] = np.full(sinp.structuralsa['i_confinement_n3'].shape, '-', dtype=object)    #bool array - This control allows confinement to occur if it is turned on for the given p6 period (controlled in feedsupply in property inputs)
    sav['n_fs_dams'] = '-'      #nut options dams
    sav['n_fs_offs'] = '-'      #nut options offs
    sav['n_initial_lw_dams'] = '-'      #number of initial lws dams - note with the current code this can only be 2 or 3
    sav['adjp_lw_initial_w1'] = np.full(sinp.structuralsa['i_adjp_lw_initial_w1'].shape, '-', dtype=object)      #initial lw adjustment dams
    sav['adjp_cfw_initial_w1'] = np.full(sinp.structuralsa['i_adjp_cfw_initial_w1'].shape, '-', dtype=object)    #initial cfw adjustment dams
    sav['adjp_fd_initial_w1'] = np.full(sinp.structuralsa['i_adjp_fd_initial_w1'].shape, '-', dtype=object)      #initial fd adjustment dams
    sav['adjp_fl_initial_w1'] = np.full(sinp.structuralsa['i_adjp_fl_initial_w1'].shape, '-', dtype=object)      #initial fl adjustment dams
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
    sav['LTW_loops_increment'] = '-'                  #SA to Increment the number of LTW loops carried out in the code. The base is 2 loops with 0 increment but if using pkl fs or ltw_adj is 0 then base is 0 loops.

    ##stock parameters
    sav['srw_c2'] = np.full(uinp.parameters['i_srw_c2'].shape, '-', dtype=object)  #SA value for srw of each c2 genotype.
    sav['cl0_c2'] = np.full(uinp.parameters['i_cl0_c2'].shape, '-', dtype=object)  #SA value for litter size genotype params.
