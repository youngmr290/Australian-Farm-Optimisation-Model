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
len_o = pinp.sheep['i_o_len']
len_r = len(sinp.f_phases())
len_s = pinp.sheep['i_s_len'] #s = shear
len_t1 = pinp.sheep['i_n_dam_sales'] + len_g0
len_t2 = pinp.sheep['i_t2_len']
len_t3 = pinp.sheep['i_t3_len']
len_V = 50  #Capital V because it is an (over) estimate to initialise the v axes that will be sliced when len_v is known.
len_x = pinp.sheep['i_x_len']
len_z = len(pinp.general['i_mask_z'])


##create dict - store sa variables in dict so they can easily be changed in the exp loop
sam = dict()
sap = dict()
saa = dict()
sat = dict()
sav = dict()
sar = dict()
sam_inp = dict()
sap_inp = dict()
saa_inp = dict()
sat_inp = dict()
sav_inp = dict()
sar_inp = dict()

############
#SAP Global#
############

##Global
sap_inp['pi']=0 #global potential intake (this increases animal intake without altering animal energy profile, to alter the energy profile use ci[1:2,...]

######
#SAM #
######
##general
sam_inp['random'] = 1.0   # SA multiplier used to tweak any random variable when debugging or checking something (after being used it is best to remove it)
sam_inp['grainp'] = 1.0   # SA multiplier for all grain prices
sam_inp['grainp_k'] = np.ones(len_k, dtype=np.float64)   # SA multiplier for grain prices for each crop

##crop
sam_inp['all_rot_yield'] = 1.0   # SA multiplier for all rotation yield

##saltbush
sam_inp['sb_growth'] = 1.0   # SA multiplier for the growth of saltbush on slp (applies to all lmus and fp)

## Annual module sensitivity variables - these need to have the same name for each pasture type
sam_inp['germ','annual']                    = 1.0                                                          # SA multiplier for germination on all lmus in all periods
sam_inp['germ','understory']                    = 1.0                                                          # SA multiplier for germination on all lmus in all periods
sam_inp['germ_l','annual']                  = np.ones((len(pinp.general['i_lmu_area'])),  dtype=np.float64)  # SA multiplier for germination on each lmus in all periods
sam_inp['germ_l','understory']                  = np.ones((len(pinp.general['i_lmu_area'])),  dtype=np.float64)  # SA multiplier for germination on each lmus in all periods
sam_inp['pgr','annual']                     = 1.0                                                          # SA multiplier for growth on all lmus in all periods
sam_inp['pgr','understory']                     = 1.0                                                          # SA multiplier for growth on all lmus in all periods
sam_inp['pgr_f','annual']                   = np.ones(len(pinp.period['i_fp_idx']),  dtype=np.float64)  # SA multiplier for growth in each feed period
sam_inp['pgr_f','understory']                   = np.ones(len(pinp.period['i_fp_idx']),  dtype=np.float64)  # SA multiplier for growth in each feed period
sam_inp['pgr_l','annual']                   = np.ones((len(pinp.general['i_lmu_area'])),  dtype=np.float64)  # SA multiplier for growth on each lmus in all periods
sam_inp['pgr_l','understory']                   = np.ones((len(pinp.general['i_lmu_area'])),  dtype=np.float64)  # SA multiplier for growth on each lmus in all periods
sam_inp['dry_dmd_decline','annual']         = 1.0                                                          # SA multiplier for the decline in digestibility of dry feed
sam_inp['dry_dmd_decline','understory']         = 1.0                                                          # SA multiplier for the decline in digestibility of dry feed
sam_inp['grn_dmd_declinefoo_f','annual']    = np.ones(len(pinp.period['i_fp_idx']),  dtype=np.float64)  # SA multiplier on decline in digestibility if green feed is not grazed (to increase FOO)
sam_inp['grn_dmd_declinefoo_f','understory']    = np.ones(len(pinp.period['i_fp_idx']),  dtype=np.float64)  # SA multiplier on decline in digestibility if green feed is not grazed (to increase FOO)
sam_inp['grn_dmd_range_f','annual']         = np.ones(len(pinp.period['i_fp_idx']),  dtype=np.float64)  # SA multiplier on range in digestibility of green feed
sam_inp['grn_dmd_range_f','understory']         = np.ones(len(pinp.period['i_fp_idx']),  dtype=np.float64)  # SA multiplier on range in digestibility of green feed
sam_inp['grn_dmd_senesce_f','annual']       = np.ones(len(pinp.period['i_fp_idx']),  dtype=np.float64)  # SA multiplier on reduction in digestibility when senescing
sam_inp['grn_dmd_senesce_f','understory']       = np.ones(len(pinp.period['i_fp_idx']),  dtype=np.float64)  # SA multiplier on reduction in digestibility when senescing
sam_inp['conservation_limit_f','annual']    = np.ones(len(pinp.period['i_fp_idx']),  dtype=np.float64)  # SA multiplier for the conservation limit in each feed period
sam_inp['conservation_limit_f','understory']    = np.ones(len(pinp.period['i_fp_idx']),  dtype=np.float64)  # SA multiplier for the conservation limit in each feed period

##livestock
sam_inp['woolp_mpg'] = 1.0                      # sa multiplier for wool price at std micron
sam_inp['salep_max'] = 1.0                      #max sale price in grid
sam_inp['salep_month_adjust_s7s9p4'] = np.ones(uinp.sheep['i_salep_months_priceadj_s7s9p4'].shape, dtype=np.float64)      #monthly sale price
sam_inp['kg'] = 1.0                             #energy efficiency of adults (zf2==1)
sam_inp['mr'] = 1.0                             #Maintenance requirement of adults (zf2==1)
sam_inp['pi'] = 1.0                             #Potential intake of adults (zf2==1)
sam_inp['LTW_dams'] = 1.0                       #adjust impact of life time wool fleece effects
sam_inp['LTW_offs'] = 1.0                       #adjust impact of life time wool fleece effects
sam_inp['pi_post'] = 1.0                        #Post loop potential intake of adults (zf2==1)
sam_inp['chill'] = 1.0                        #intermediate sam on chill.

##stock parameters
sam_inp['ci_c2'] = np.ones(uinp.parameters['i_ci_c2'].shape, dtype=np.float64)  #intake params for genotypes
sam_inp['sfw_c2'] = 1.0                         #std fleece weight genotype params
sam_inp['rr'] = 1.0                        #scanning percentage (adjust the standard scanning % for f_conception_ltw and within function for f_conception_cs
sam_inp['husb_cost_h2'] = np.ones(uinp.sheep['i_husb_operations_contract_cost_h2'].shape, dtype=np.float64)  #SA value for contract cost of husbandry operations.
sam_inp['husb_mustering_h2'] = np.ones(uinp.sheep['i_husb_operations_muster_propn_h2'].shape, dtype=np.float64)  #SA value for mustering required for husbandry operations.
sam_inp['husb_labour_l2h2'] = np.ones(uinp.sheep['i_husb_operations_labourreq_l2h2'].shape, dtype=np.float64)  #units of the job carried out per husbandry labour hour

######
#SAP #
######
sap_inp['evg'] = 0.0               #energy content of liveweight gain - this is a high level sa, it impacts within a calculation not on an input and is only implemented on adults
sap_inp['mortalityp'] = 0.0        #Scale the calculated progeny mortality at birth relative - this is a high level sa, it impacts within a calculation not on an input
sap_inp['mortalitye'] = 0.0        #Scale the calculated dam mortality at birth - this is a high level sa, it impacts within a calculation not on an input
sap_inp['mortalityb'] = 0.0        #Scale the calculated base mortality - this is a high level sa, it impacts within a calculation not on an input
sap_inp['kg_post'] = 0.0           #Post loop energy efficiency of adults (zf2==1)
sap_inp['mr_post'] = 0.0           #Post loop maintenance requirement of adults (zf2==1)


######
#SAA #
######
##pasture
saa_inp['germ','annual']                    = 0.0                                                          # SA addition for germination on all lmus in all periods
saa_inp['germ','understory']                    = 0.0                                                          # SA addition for germination on all lmus in all periods
saa_inp['pgr','annual']                     = 0.0                                                          # SA addition for growth on all lmus in all periods
saa_inp['pgr','understory']                     = 0.0                                                          # SA addition for growth on all lmus in all periods
saa_inp['pgr_f','annual']                   = np.zeros(len(pinp.period['i_fp_idx']),  dtype=np.float64)  # SA addition for growth in each feed period
saa_inp['pgr_f','understory']                   = np.zeros(len(pinp.period['i_fp_idx']),  dtype=np.float64)  # SA addition for growth in each feed period
saa_inp['pgr_l','annual']                   = np.zeros((len(pinp.general['i_lmu_area'])),  dtype=np.float64)  # SA addition for growth on each lmus in all periods
saa_inp['pgr_l','understory']                   = np.zeros((len(pinp.general['i_lmu_area'])),  dtype=np.float64)  # SA addition for growth on each lmus in all periods

##stock
saa_inp['husb_cost_h2'] = np.zeros(uinp.sheep['i_husb_operations_contract_cost_h2'].shape, dtype=np.float64)  #SA value for contract cost of husbandry operations.
saa_inp['husb_labour_l2h2'] = np.zeros(uinp.sheep['i_husb_operations_labourreq_l2h2'].shape, dtype=np.float64)  #units of the job carried out per husbandry labour hour
saa_inp['r1_izg1'] = np.zeros(pinp.sheep['ia_r1_zig1'].shape, dtype=int)   #SA to change the base feed option selected for dams
saa_inp['r1_izg3'] = np.zeros(pinp.sheep['ia_r1_zig3'].shape, dtype=int)   #SA to change the base feed option selected for offspring
saa_inp['r2_isk2g1'] = np.zeros(pinp.sheep['ia_r2_isk2g1'].shape, dtype=int)   #SA to change the base feed option selected for dams
saa_inp['r2_ik5g3'] = np.zeros(pinp.sheep['ia_r2_ik5g3'].shape, dtype=int)   #SA to change the base feed option selected for offspring
saa_inp['date_born1st_iog'] = np.zeros(pinp.sheep['i_date_born1st_iog2'].shape, dtype=int)  #SA to adjust lambing date (used for ewe lambs).
saa_inp['feedsupply_r1jp'] = np.zeros(pinp.feedsupply['i_feedsupply_options_r1j2p'].shape, dtype=np.float64)  #SA value for feedsupply.
saa_inp['feedsupply_adj_r2p'] = np.zeros(pinp.feedsupply['i_feedsupply_adj_options_r2p'].shape, dtype=np.float64)  #SA value for feedsupply adjustment.

##stock parameters
saa_inp['sfd_c2'] = 0.0                     #std fibre diameter genotype params
saa_inp['cl0_c2'] = np.zeros(uinp.parameters['i_cl0_c2'].shape, dtype=np.float64)  #SA value for litter size genotype params.
saa_inp['scan_std_c2'] = 0.0                #std scanning percentage of a genotype. Controls the MU repro, initial propn of sing/twin/trip prog required to replace the dams, the lifetime productivity of the dams as affected by their BTRT..
saa_inp['nlb_c2'] = 0.0                #std scanning percentage of a genotype. Controls the MU repro, initial propn of sing/twin/trip prog required to replace the dams, the lifetime productivity of the dams as affected by their BTRT..
saa_inp['rr'] = 0.0                    #reproductive rate/scanning percentage (adjust the standard scanning % for f_conception_ltw and within function for f_conception_cs
saa_inp['rr_age_og1'] = np.zeros(pinp.sheep['i_scan_og1'].shape, dtype=np.float64)    # reproductive rate by age. Use shape that has og1
saa_inp['mortalityx'] = np.zeros(np.max(sinp.stock['a_nfoet_b1'])+1, dtype=np.float64)  #Adjust the progeny mortality due to exposure at birth relative - this is a high level sa, it impacts within a calculation not on an input
saa_inp['wean_wt'] = 0.0            #weaning weight adjustment of yatf. Note: WWt changes without any change in MEI

######
#SAT #
######
sat_inp['salep_weight_scalar'] = 0.0 #Scalar for LW impact across grid 1
sat_inp['salep_score_scalar'] = 0.0  #Scalar for score impact across the grid


######
#SAR #
######


######
#SAV #
######
##if you initialise an array it must be type object (so that you can assign int/float/bool into the array)
sav_inp['feedsupply_adj_r2p'] = np.full_like(pinp.feedsupply['i_feedsupply_adj_options_r2p'], '-', dtype=object)  #SA value for feedsupply adjustment.

##general
sav_inp['steady_state']      = '-'                  #SA to alter if the model is steady state
sav_inp['mask_z']      = np.full_like(pinp.general['i_mask_z'], '-', dtype=object)   #SA to alter which seasons are included
sav_inp['inc_node_periods']      = '-'              #SA to alter if season nodes are included in the steady state model (note they are always included in the dsp version this only effects if they are included in steady state)
sav_inp['seq_len']      = '-'                     #SA to alter the length of the season sequence in the SQ model
sav_inp['rev_create']      = '-'                  #SA to alter if the trial is being used to create rev std values
sav_inp['rev_number']      = '-'                  #SA to alter rev number - rev number is appended to the std rev value pkl file and can be used to select which rev is used as std for a given trial.
sav_inp['rev_trait_inc'] = np.full_like(sinp.structuralsa['i_rev_trait_inc'], '-', dtype=object) #SA value for which traits are to be held constant in REV analysis.
sav_inp['fs_create_pkl']      = '-'                  #SA to control if the trial is being used to create pkl fs
sav_inp['fs_create_number']      = '-'                  #SA to alter fs number - fs number is appended to the fs pkl file and can be used to select which pkl fs is created for a given trial.
sav_inp['gen_with_t']      = '-'                  #SA to control if sheep generator is run with active t axis.
sav_inp['fs_use_pkl']      = '-'                  #SA to control if the pkl fs is used or the excel input fs is used.
sav_inp['fs_use_number']      = '-'                  #SA to alter fs number - fs number is appended to the fs pkl file and can be used to select which pkl fs is used for a given trial.
sav_inp['r2adjust_inc']      = '-'              #SA to control if the r2 feedsupply adjustment from Excel is included.
sav_inp['inc_c1_variation'] = '-'               #control if price variation is on. This only effects result if risk aversion is included.
sav_inp['inc_risk_aversion'] = '-'              #control if risk aversion is included. Default is not included (ie utility=profit).
sav_inp['utility_method'] = '-'              #control which utility function is used
sav_inp['cara_risk_coef'] = '-'              #control risk coefficient for CRRA method
sav_inp['crra_risk_coef'] = '-'              #control risk coefficient for CRRA method
sav_inp['pinp_rot'] = '-'                       #control if using the pinp rotations or the full rotation list (note full rot requires simulation inputs)
sav_inp['mach_option'] = '-'                    #control which machine compliment is used
sav_inp['lmu_area_l']    = np.full(len(pinp.general['i_lmu_area']), '-', dtype=object)  # SA for area of each LMU

##finance
sav_inp['minroe']      = '-'                  #SA to alter the minroe (applied to both steady-state and dsp minroe inputs)
sav_inp['overdraw_limit']      = '-'          #SA to alter the overdraw limit (amount of money that can be loaned from bank)
sav_inp['interest_rate']      = '-'           #SA to alter the credit and debit interest from bank
sav_inp['opp_cost_capital']      = '-'        #SA to alter the opportunity cost of capital

##price
sav_inp['grain_percentile'] = '-'  #grain price percentile

##labour
sav_inp['casual_ub'] = '-'  #casual upper bound all year except seeding and harv
sav_inp['seedharv_casual_ub'] = '-'  #casual upper bound at seeding and harv

##cropgrazing
sav_inp['cropgrazing_inc'] = '-'  #control if crop grazing is allowed

##salt land pasture
sav_inp['slp_inc'] = '-'  #control if salt land pasture is included

##bounds
sav_inp['bnd_slp_area_l'] = np.full(len_l, '-', dtype=object)  #control the area of slp on each lmu
sav_inp['bnd_sb_consumption_p6'] = np.full(len(pinp.period['i_fp_idx']), '-', dtype=object)  #upper bnd on the amount of sb consumed
sav_inp['bnd_total_pas_area'] = '-'  #Total pasture area for bound. '-' is default so it will chuck an error if the bound is turned on without a specified area
sav_inp['bnd_pasarea_inc'] = '-'   #SA to turn on the pasture area bound
sav_inp['bnd_rotn_inc'] = '-'   #SA to turn on the phase area bounds
sav_inp['bnd_sr_inc'] = '-'   #SA to turn on the stocking rate bounds
sav_inp['bnd_propn_dams_mated_og1'] = np.full((len_d,) + pinp.sheep['i_g3_inc'].shape, '-', dtype=object)   #proportion of dams mated
sav_inp['est_propn_dams_mated_og1'] = np.full((len_d,) + pinp.sheep['i_g3_inc'].shape, '-', dtype=object)   #estimated proportion of dams mated - used when bnd_propn is default "-"
sav_inp['bnd_drys_sold_o'] = np.full(pinp.sheep['i_dry_sales_forced_o'].shape, '-', dtype=object)   #SA to force drys to be sold
sav_inp['bnd_drys_retained_o'] = np.full(pinp.sheep['i_dry_retained_forced_o'].shape, '-', dtype=object)   #SA to force drys to be retained
sav_inp['est_drys_retained_scan_o'] = np.full(pinp.sheep['i_drys_retained_scan_est_o'].shape, '-', dtype=object)   #Estimate of the propn of drys sold at scanning
sav_inp['est_drys_retained_birth_o'] = np.full(pinp.sheep['i_drys_retained_birth_est_o'].shape, '-', dtype=object)   #Estimate of the propn of drys sold at birth
sav_inp['bnd_sale_twice_dry_inc'] = '-'   #SA to include the bound which forces twice dry dams to be sold
sav_inp['bnd_twice_dry_propn'] = '-'   #SA to change twice dry dam proportion
sav_inp['bnd_lo_dam_inc'] = '-'   #control if dam lower bound is on.
sav_inp['bnd_lo_dams_tog1'] = np.full((len_t1,) + (len_d,) + (len_g1,), '-', dtype=object)   #min number of dams
sav_inp['bnd_lo_dams_tVg1'] = np.full((len_t1,) + (len_V,) + (len_g1,), '-', dtype=object)   #min number of dams
sav_inp['bnd_up_dam_inc'] = '-'   #control if dam upper bound is on.
sav_inp['bnd_up_dams_tog1'] = np.full((len_t1,) + (len_d,) + (len_g1,), '-', dtype=object)   #max number of dams
sav_inp['bnd_up_dams_tVg1'] = np.full((len_t1,) + (len_V,) + (len_g1,), '-', dtype=object)   #max number of dams
sav_inp['bnd_total_dams_scanned'] = '-'   #total dams scanned (summed over all dvps) - this also controls if bound is on.
sav_inp['bnd_propn_dam5_retained'] = '-'   #propn of 5yo dams retained - this also controls if bound is on.
sav_inp['bnd_lo_off_inc'] = '-'   #control if off lower bound is on.
sav_inp['bnd_lo_offs_tsdxg3'] = np.full((len_t3,) + (len_s,) + (len_d,) + (len_x,) + (len_g3,), '-', dtype=object)   #min number of offs
sav_inp['bnd_up_off_inc'] = '-'   #control if off upper bound is on.
sav_inp['bnd_up_offs_tsdxg3'] = np.full((len_t3,) + (len_s,) + (len_d,) + (len_x,) + (len_g3,), '-', dtype=object)   #max number of offs
sav_inp['bnd_up_prog_inc'] = '-'   #control if prog upper bound is on.
sav_inp['bnd_up_prog_tdxg2'] = np.full((len_t2,) + (len_d,) + (len_x,) + (len_g2,), '-', dtype=object)   #max number of offs
sav_inp['bnd_sr_t'] = np.full(pinp.sheep['i_sr_constraint_t'].shape, '-', dtype=object)   #SA to fix stocking rate
sav_inp['bnd_min_sale_age_wether_g3'] = np.full(pinp.sheep['i_g3_inc'].shape, '-', dtype=object)   #SA to set min age wether can be sold
sav_inp['bnd_max_sale_age_wether_g3'] = np.full(pinp.sheep['i_g3_inc'].shape, '-', dtype=object)   #SA to set max age wether can be sold
sav_inp['bnd_min_sale_age_female_g1'] = np.full(pinp.sheep['i_g3_inc'].shape, '-', dtype=object)   #SA to set min age a dam can be sold - BBT offspring can be sold but BBT dams can't (because they are BB)
sav_inp['bnd_min_sale_age_female_dg3'] = np.full((len_d,) + (len_g3,), '-', dtype=object)   #SA to set min age a female can be sold - used to bound prog & offs
sav_inp['bnd_max_sale_age_female_g3'] = np.full(pinp.sheep['i_g3_inc'].shape, '-', dtype=object)   #SA to set max age wether can be sold
sav_inp['rot_lobound_rl'] = np.full((len_r,) + (len_l,), '-', dtype=object)

##pasture
sav_inp['pas_inc_t'] = np.full_like(pinp.general['pas_inc'], '-', dtype=object) #SA value for pastures included mask

##Stock
sav_inp['nv_inc'] = '-'    #SA to store NV report values
sav_inp['lw_inc'] = '-'     #SA to store LW report values
sav_inp['ffcfw_inc'] = '-'  #SA to store FFCFW report values
sav_inp['onhand_mort_p_inc'] = '-'  #SA to store onhand report values
sav_inp['mort_inc'] = '-'  #SA to store mort report values
sav_inp['feedbud_inc'] = '-'  #SA to store feed budget report values
sav_inp['eqn_compare']      = '-'                  #SA to alter if the different equation systems in the sheep sim are run and compared
sav_inp['eqn_used_g0_q1p7'] = np.full(uinp.sheep['i_eqn_used_g0_q1p7'].shape, '-', dtype=object) #SA value for which equation system to use
sav_inp['eqn_used_g1_q1p7'] = np.full(uinp.sheep['i_eqn_used_g1_q1p7'].shape, '-', dtype=object) #SA value for which equation system to use
sav_inp['eqn_used_g2_q1p7'] = np.full(uinp.sheep['i_eqn_used_g2_q1p7'].shape, '-', dtype=object) #SA value for which equation system to use
sav_inp['eqn_used_g3_q1p7'] = np.full(uinp.sheep['i_eqn_used_g3_q1p7'].shape, '-', dtype=object) #SA value for which equation system to use
sav_inp['TOL_inc']          = np.full(pinp.sheep['i_mask_i'].shape, '-', dtype=object)      # SA value for the inclusion of each TOL
sav_inp['g3_included']      = np.full(pinp.sheep['i_g3_inc'].shape, '-', dtype=object)      # SA value for the inclusion of each offspring genotype
sav_inp['genotype']         = np.full(pinp.sheep['a_c2_c0'].shape, '-', dtype=object)       # this is the selection of the genotypes of the sires for B, M & T
sav_inp['scan_og1']         = np.full(pinp.sheep['i_scan_og1'].shape, '-', dtype=object)    # SA value for the scanning management option
sav_inp['woolp_mpg_percentile'] = '-'               #sa value for the wool price percentile
sav_inp['woolp_mpg'] = '-'                          # sa value for wool price at std micron
sav_inp['woolp_fdprem_percentile'] = '-'            # sa value for fd premium percentile (premium received by fd compared to std)
sav_inp['woolp_fdprem'] = '-'                       # sa value for fd premium
sav_inp['salep_percentile'] = '-'                   #Value for percentile for all sale grids
sav_inp['salep_max'] = '-'                          #max sale price in grid
len_max_w1 = sinp.structuralsa['i_w_start_len1'] * len(sinp.structuralsa['i_nut_spread_n1']) ** (
        len(sinp.stock['i_fixed_fvp_mask_dams'])+len(sinp.structuralsa['i_fvp_mask_dams'])) #the max size of w if all n and fvps included.
len_max_w3 = sinp.structuralsa['i_w_start_len3'] * len(sinp.structuralsa['i_nut_spread_n3']) ** len(sinp.structuralsa['i_fvp_mask_offs']) #the max size of w if all n and fvps included.
sav_inp['nut_mask_dams_oWi'] = np.full((pinp.sheep['i_o_len'], len_max_w1, pinp.sheep['i_i_len']), '-', dtype=object)    #masks the nutrition options available e.g. high low high - the options selected are available for each starting weight (ie len_W = len_w/n_start_weights). This array is cut down in the code to the correct w len.
sav_inp['nut_mask_offs_sWix'] = np.full((pinp.sheep['i_s_len'], len_max_w3, pinp.sheep['i_i_len'], pinp.sheep['i_x_len']), '-', dtype=object)   #masks the nutrition options available e.g. high low high - the options selected are available for each starting weight (ie len_W = len_w/n_start_weights). This array is cut down in the code to the correct w len.
sav_inp['nut_spread_n1'] = np.full(sinp.structuralsa['i_nut_spread_n1'].shape, '-', dtype=object)      #nut spread dams
sav_inp['nut_spread_n3'] = np.full(sinp.structuralsa['i_nut_spread_n3'].shape, '-', dtype=object)      #nut spread offs
sav_inp['n_fs_dams'] = '-'      #nut options dams
sav_inp['n_fs_offs'] = '-'      #nut options offs
sav_inp['n_initial_lw_dams'] = '-'      #number of initial lws dams - note with the current code this can only be 2 or 3
sav_inp['adjp_lw_initial_w1'] = np.full(sinp.structuralsa['i_adjp_lw_initial_w1'].shape, '-', dtype=object)      #initial lw adjustment dams
sav_inp['adjp_cfw_initial_w1'] = np.full(sinp.structuralsa['i_adjp_cfw_initial_w1'].shape, '-', dtype=object)    #initial cfw adjustment dams
sav_inp['adjp_fd_initial_w1'] = np.full(sinp.structuralsa['i_adjp_fd_initial_w1'].shape, '-', dtype=object)      #initial fd adjustment dams
sav_inp['adjp_fl_initial_w1'] = np.full(sinp.structuralsa['i_adjp_fl_initial_w1'].shape, '-', dtype=object)      #initial fl adjustment dams
sav_inp['mask_fvp_dams'] = np.full(sinp.structuralsa['i_fvp_mask_dams'].shape, '-', dtype=object)      #SA to mask optional fvps.
sav_inp['fvp_is_dvp_dams'] = np.full(sinp.structuralsa['i_dvp_mask_f1'].shape, '-', dtype=object)      #SA to control if optional fvp is a dvp (note: fvps don't need to be dvps, the only benefit is if new information is available e.g. if animals uncluster, which allows differential management).
sav_inp['mask_fvp_offs'] = np.full(sinp.structuralsa['i_fvp_mask_offs'].shape, '-', dtype=object)      #SA to mask optional fvps.
sav_inp['fvp_is_dvp_offs'] = np.full(sinp.structuralsa['i_fvp_mask_offs'].shape, '-', dtype=object)      #SA to control if optional fvp is a dvp (note: fvps don't need to be dvps, the only benefit is if new information is available e.g. if animals uncluster, which allows differential management).
sav_inp['r1_izg1'] = np.full(pinp.sheep['ia_r1_zig1'].shape, '-', dtype=object)   #SA to change the base feed option selected for dams
sav_inp['r1_izg3'] = np.full(pinp.sheep['ia_r1_zig3'].shape, '-', dtype=object)   #SA to change the base feed option selected for offspring
sav_inp['r2_ik0g1'] = np.full(pinp.sheep['ia_r2_ik0g1'].shape, '-', dtype=object)   #SA to change the selected feed adjustments selected for the k0 axis (wean age) for dams
sav_inp['r2_ik0g3'] = np.full(pinp.sheep['ia_r2_ik0g3'].shape, '-', dtype=object)   #SA to change the selected feed adjustments selected for the k0 axis (wean age) for offs
sav_inp['r2_isk2g1'] = np.full(pinp.sheep['ia_r2_isk2g1'].shape, '-', dtype=object)   #SA to change the selected feed adjustments selected for the k2 axis (LSLN) for dams
sav_inp['r2_ik5g3'] = np.full(pinp.sheep['ia_r2_ik5g3'].shape, '-', dtype=object)   #SA to change the selected feed adjustments selected for the k5 axis (BTRT) for offs
sav_inp['period_is_reportffcfw_p'] = np.full(500, '-', dtype=object)   #todo remove after ewelamb analysis
sav_inp['LTW_loops'] = '-'                  #SA to control the number of ltw loops. Default is 1 (ie ltw adjustment not included).

##stock parameters
sav_inp['srw_c2'] = np.full(uinp.parameters['i_srw_c2'].shape, '-', dtype=object)  #SA value for srw of each c2 genotype.
sav_inp['cl0_c2'] = np.full(uinp.parameters['i_cl0_c2'].shape, '-', dtype=object)  #SA value for litter size genotype params.
