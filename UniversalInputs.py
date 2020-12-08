# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 16:06:06 2019

module: universal module - contains all the core input data - usually held constant/doesn't change between regions or farms'


Version Control:
Version     Date        Person  Change
1.1         25Dec19     John    structure['phase_len'] = 5 (rather than 4)
1.2         27Dec19     MRY     moved rotation input data from crop to here
1.3         13Jan20     MRY     changed input.py to universal - and added other bits such as price, interest rates and mach options
1.4         24Feb20     MRY     Added the capital set names to the set definition - this is required to build the pasture germination df without to many loops

Known problems:
Fixed   Date    ID by   Problem
1.2     25Dec19 John    The phase description in inputs are the full word whereas in the rotation phase it is just the letter.

@author: young
"""

##python modules
import pickle as pkl
import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta

import Functions as fun


#########################################################################################################################################################################################################
#########################################################################################################################################################################################################
#read in excel
#########################################################################################################################################################################################################
#########################################################################################################################################################################################################
import os.path
try:
    if os.path.getmtime("Universal.xlsx") > os.path.getmtime("pkl_universal.pkl"):
        inputs_from_pickle = False 
    else: 
        inputs_from_pickle = True
        print( 'Reading universal inputs from pickle')
except FileNotFoundError:      
    inputs_from_pickle = False

filename= 'pkl_universal.pkl'
##if inputs are not read from pickle then they are read from excel and written to pickle
if inputs_from_pickle == False:
    with open(filename, "wb") as f:
        ##prices
        price_inp = fun.xl_all_named_ranges("Universal.xlsx","Price")
        pkl.dump(price_inp, f)
        
        ##Finance inputs
        finance_inp = fun.xl_all_named_ranges("Universal.xlsx","Finance")
        pkl.dump(finance_inp, f)
        
        ##mach inputs - general
        mach_general_inp = fun.xl_all_named_ranges("Universal.xlsx","Mach General")
        pkl.dump(mach_general_inp, f)
        
        ##feed inputs
        feed_inputs_inp = fun.xl_all_named_ranges("Universal.xlsx","Feed Budget")
        pkl.dump(feed_inputs_inp, f)
        
        ##sup inputs
        sup_inp = fun.xl_all_named_ranges("Universal.xlsx","Sup Feed")
        pkl.dump(sup_inp, f)
        
        ##crop inputs
        crop_inp = fun.xl_all_named_ranges("Universal.xlsx","Crop Sim")
        pkl.dump(crop_inp, f)
        
        ##sheep inputs
        sheep_inp = fun.xl_all_named_ranges('Inputs parameters.xlsm', ['Universal'], numpy=True)
        pkl.dump(sheep_inp, f)
        parameters_inp = fun.xl_all_named_ranges('Inputs parameters.xlsm', ['Parameters'], numpy=True)
        pkl.dump(parameters_inp, f)
        pastparameters_inp = fun.xl_all_named_ranges('Inputs parameters.xlsm', ['PastParameters'], numpy=True)
        pkl.dump(pastparameters_inp, f)
        
        ##mach options
        ###create a dict to store all options - this allows the user to select an option
        machine_options_dict_inp={}
        machine_options_dict_inp[1] = fun.xl_all_named_ranges("Universal.xlsx","Mach 1")
        pkl.dump(machine_options_dict_inp, f)

##else the inputs are read in from the pickle file
##note this must be in the same order as above
else:
    with open(filename, "rb") as f:
        price_inp = pkl.load(f)
        
        finance_inp = pkl.load(f)
        
        mach_general_inp = pkl.load(f)
        
        feed_inputs_inp = pkl.load(f)
        
        sup_inp = pkl.load(f)
        
        crop_inp = pkl.load(f)
        
        sheep_inp = pkl.load(f)
        
        parameters_inp = pkl.load(f)
        pastparameters_inp = pkl.load(f)
        
        machine_options_dict_inp  = pkl.load(f)
        
        
        
price = price_inp.copy()
finance = finance_inp.copy()
mach_general = mach_general_inp.copy()
feed_inputs = feed_inputs_inp.copy()
supfeed = sup_inp.copy()
crop = crop_inp.copy()
sheep = sheep_inp.copy()
parameters = parameters_inp.copy()
pastparameters = pastparameters_inp.copy()
mach = machine_options_dict_inp.copy()

#######################
#apply SA             #
#######################
def univeral_inp_sa():
    '''
    
    Returns
    -------
    None.
    
    Applies sensitivity adjustment to each input.
    This function gets called at the beginning of each loop in the exp.py module

    '''
    ##have to import it here since sen.py imports this module
    import Sensitivity as sen 
    ##enter sa below

    ##sheep
    ###SAT
    sheep['i_salep_weight_scalar_s7s5s6'] = fun.f_sa(sheep['i_salep_weight_scalar_s7s5s6'], sen.sat['salep_weight_scalar'], 3, 1, 0) #Scalar for LW impact across grid 1 (sat adjusted)
    sheep['i_salep_score_scalar_s7s5s6'] = fun.f_sa(sheep['i_salep_score_scalar_s7s5s6'], sen.sat['salep_score_scalar'], 3, 1, 0) #Scalar for score impact across the grid (sat adjusted)
    ###SAV
    sheep['i_eqn_compare'] = fun.f_sa(sheep_inp['i_eqn_compare'], sen.sav['eqn_compare'], 5)
    sheep['i_woolp_mpg_percentile'] = fun.f_sa(sheep['i_woolp_mpg_percentile'], sen.sav['woolp_mpg_percentile'], 5) #replaces the std percentile input with the sa value
    sheep['i_woolp_fdprem_percentile'] = fun.f_sa(sheep['i_woolp_fdprem_percentile'], sen.sav['woolp_fdprem_percentile'], 5) #replaces the std percentile input with the sa value
    sheep['i_salep_percentile'] = fun.f_sa(sheep['i_salep_percentile'], sen.sav['salep_percentile'], 5) #Value for percentile for all sale grids



    
    
    

#########################################################################################################################################################################################################
#########################################################################################################################################################################################################
#general - used to determine model structure (these will stay in python to keep seperate from excel inputs which can be adjusted by any user)
#########################################################################################################################################################################################################
#########################################################################################################################################################################################################

##create an empty dict to store all structure inputs
structure = dict()

###############
# labour      #
###############
structure['worker_levels'] = ['anyone', 'perm', 'manager']

###############
# crop        #
###############
##grain pools there is one transfer constraint for each pool.
structure['grain_pools']=['firsts','seconds']

###############
# cashflow    #
###############
##asset value time of yr - this is also the begining of the cashflow periods  ^for now this must be the 1/1/19 but it would be good to make it flexible ie have the capacity to have cashflow periods start on any day of yr
structure['i_date_assetvalue']= datetime.datetime(2019, 1, 1) #y/m/d

##the number of these can change as long as each period is of equal length.
structure['cashflow_periods']=['JF$FLOW','MA$FLOW','MJ$FLOW','JA$FLOW','SO$FLOW','ND$FLOW']

###############
# pasture     #
###############
##sets as well as define the pastures to include
structure['pastures'] = ['annual'] # ,'lucerne','tedera']    #define which pastures are to be included
structure['dry_groups'] = ['Dry-L', 'Dry-H']                       # Low & high quality groups for dry feed
structure['grazing_int'] =  ['Graz0', 'Graz25', 'Graz50', 'Graz100']   # grazing intensity in the growth/grazing activities
structure['foo_levels'] =  ['Foo-L', 'Foo-M', 'Foo-H']                 # Low, medium & high FOO level in the growth/grazing activities

#######
#sheep#
#######
##dse
structure['ia_sire_dsegroup'] = 4
structure['ia_offs_dsegroup'] = 4
structure['ia_dams_dsegroup_b1'] = np.array([4,	0,	1,	2,	3,	1,	2,	1,	0,	0,	0])
##axis pos ^maybe move in the ones from the other inputs to here?
structure['i_e0_pos']=-5
##general
structure['i_age_max'] = 7.1  # after shearing for the July lambing at 6.5 yo
structure['i_age_max_offs'] = 3.5
structure['i_sim_periods_year'] = 52
structure['i_w_pos'] = -10
structure['i_n_pos'] = -11
structure['i_p_pos'] = -15
structure['i_k2_pos'] = -17
structure['i_k3_pos'] = -18
structure['i_k5_pos'] = -17
structure['i_lag_wool'] = 1 #lags in calculations (number of days over which production is averaged)
structure['i_lag_organs'] = 1  #lags in calculations (number of days over which production is averaged)
structure['i_lsln_idx_dams'] = ['NM', '00',	'11',	'22',	'33',	'21',	'32',	'31',	'10',	'20',	'30']
structure['i_btrt_idx_offs'] = ['11',	'22',	'33',	'21',	'32',	'31']
structure['prejoin_offset'] = 8
structure['i_feedsupply_itn_max'] = 10

##pools
structure['sheep_pools']=['pool0', 'pool1', 'pool2', 'pool3'] #nutrition pools
## DSE group and LSLN (b1)
structure['i_mask_b0_b1'] = np.array([False, False, True,	True,	True,	True,	True,	True,	False,	False,	False])
structure['i_mated_b1'] = np.array([False, True, True,	True,	True,	True,	True,	True,	True,	True,	True])
structure['ia_b0_b1'] = np.array([0, 0, 0, 1,	2,	3,	4,	5,	0,	0,	0])
structure['a_prepost_b1'] = np.array([0, 1, 2, 3, 4, 3,	4,	4,	2,	3,	4]) #The association of b1 pre lambing pointed to from b1 post lambing
structure['a_nfoet_b1'] = np.array([0,0,1,2,3,2,3,3,1,2,3])
structure['a_nyatf_b1'] = np.array([0,0,1,2,3,1,2,1,0,0,0])
structure['i_initial_b1'] = np.array([1,0,0,0,0,0,0,0,0,0,0])
structure['i_numbers_min_b1'] = np.array([0,0,0,0,0,0,0,0,0,0,0])

                                        #dams
##dams sire transfer                #bbb bbm bbt bmt
structure['ia_g1_tg1'] = np.array([                 #sire
                                    [0,	0,	0,	4],  #b   use 4 - A value greater than the number of slices of the g axis, so that a_g1_tg1 == index_g1 is never True
                                    [1,	1,	1,	4],  #m
                                    [2,	2,	2,	3]]) #t

                                               #dams
##dams sire transfer  mask           #bbb   bbm     bbt     bmt
structure['i_transfer_exists_tg1'] = np.array([                    #sire
                                    [True,	True,	True,	False],  #b
                                    [True,	True,	True,	False],  #m
                                    [True,	True,	True,	True]])  #t

##feed supply/ nutrition levels
structure['i_w0_len'] = 1
structure['i_w_idx_sire'] = ['lw0']
structure['i_w1_len'] = 81
structure['i_w_idx_dams'] = ['lw0', 'lw1', 'lw2', 'lw3', 'lw4', 'lw5', 'lw6', 'lw7', 'lw8', 'lw9', 'lw10', 'lw11', 'lw12', 'lw13', 'lw14', 'lw15', 'lw16', 'lw17', 'lw18', 'lw19', 'lw20', 'lw21', 'lw22', 'lw23', 'lw24', 'lw25', 'lw26', 'lw27', 'lw28', 'lw29', 'lw30', 'lw31', 'lw32', 'lw33', 'lw34', 'lw35', 'lw36', 'lw37', 'lw38', 'lw39', 'lw40', 'lw41', 'lw42', 'lw43', 'lw44', 'lw45', 'lw46', 'lw47', 'lw48', 'lw49', 'lw50', 'lw51', 'lw52', 'lw53', 'lw54', 'lw55', 'lw56', 'lw57', 'lw58', 'lw59', 'lw60', 'lw61', 'lw62', 'lw63', 'lw64', 'lw65', 'lw66', 'lw67', 'lw68', 'lw69', 'lw70', 'lw71', 'lw72', 'lw73', 'lw74', 'lw75', 'lw76', 'lw77', 'lw78', 'lw79', 'lw80']
structure['i_progeny_w2_len'] = 10
structure['i_w3_len'] = 81
structure['i_w_idx_offs'] = ['lw0', 'lw1', 'lw2', 'lw3', 'lw4', 'lw5', 'lw6', 'lw7', 'lw8', 'lw9', 'lw10', 'lw11', 'lw12', 'lw13', 'lw14', 'lw15', 'lw16', 'lw17', 'lw18', 'lw19', 'lw20', 'lw21', 'lw22', 'lw23', 'lw24', 'lw25', 'lw26', 'lw27', 'lw28', 'lw29', 'lw30', 'lw31', 'lw32', 'lw33', 'lw34', 'lw35', 'lw36', 'lw37', 'lw38', 'lw39', 'lw40', 'lw41', 'lw42', 'lw43', 'lw44', 'lw45', 'lw46', 'lw47', 'lw48', 'lw49', 'lw50', 'lw51', 'lw52', 'lw53', 'lw54', 'lw55', 'lw56', 'lw57', 'lw58', 'lw59', 'lw60', 'lw61', 'lw62', 'lw63', 'lw64', 'lw65', 'lw66', 'lw67', 'lw68', 'lw69', 'lw70', 'lw71', 'lw72', 'lw73', 'lw74', 'lw75', 'lw76', 'lw77', 'lw78', 'lw79', 'lw80']
structure['i_n0_len'] = 1  #number of different feedsupplies in each fv period
structure['i_n_idx_sire'] = ['n1']
structure['i_n1_len'] = 3   #number of different feedsupplies in each fv period
structure['i_n_idx_dams'] = ['n1']
structure['i_n3_len'] = 3  #number of different feedsupplies in each fv period
structure['i_n_idx_offs'] = ['n1']
structure['i_n0_matrix_len'] = 1 #number of nutrition levels in the matrix
structure['i_n1_matrix_len'] = 1 #number of nutrition levels in the matrix
structure['i_n3_matrix_len'] = 1 #number of nutrition levels in the matrix
structure['i_n_fvp_period0'] = 1 #number of different fs period
structure['i_n_fvp_period1'] = 3 #number of different fs period
structure['i_n_fvp_period3'] = 3 #number of different fs period

structure['i_nut_spread_n0'] = np.array([0])
structure['i_nut_spread_n1'] = np.array([0,1,-1]) #fs adjustment for different n levels - above 3 is absolute not adjustemnt
structure['i_density_g1_n'] = np.array([1,0.5,1.5]) #stocking density adjuster for different n levels. An increasing feedsupply (less than 3.0) means that the animals are being offered more feed and therefore density is lower (although it could be with a high density and lots of supplement - we will be assuming that it is lower density and increased FOO). This is represented by scaling the standard stocking density by a number less than 1. Note: Distance walked is scaled by 40/density (if density is > 40). SO trying to make distance a small number for confinement feeding and even smaller for feedlotting
structure['i_nut_spread_n3'] = np.array([0,1,-1]) #fs adjustment for different n levels - above 3 is absolute not adjustemnt
structure['i_density_g3_n'] = np.array([1,0.5,1.5]) #stocking density adjuster for different n levels. An increasing feedsupply (less than 3.0) means that the animals are being offered more feed and therefore density is lower (although it could be with a high density and lots of supplement - we will be assuming that it is lower density and increased FOO). This is represented by scaling the standard stocking density by a number less than 1. Note: Distance walked is scaled by 40/density (if density is > 40). SO trying to make distance a small number for confinement feeding and even smaller for feedlotting
##genotype
###An array that contains the proportion of each purebred genotype in the sire, dam, yatf or offspring eg:
# 		            k0	
# g3		 B	     M	    T	
# B		    1.0			
# BM		0.5	    0.5		
# BT		0.5		0.5	
# BMT		0.25	0.25	0.5	

structure['i_mul_g0c0'] = np.array([[1,0,0],
                                     [0,1,0],
                                     [0,0,1]])    
structure['i_mul_g1c0'] = np.array([[1,   0,    0],
                                     [1,   0,    0],    
                                     [1,   0,    0],    
                                     [0.5, 0.5,  0]])    
structure['i_mul_g2c0'] = np.array([[1,   0,    0],
                                     [0.5,  0.5,  0],
                                     [0.5,  0,    0.5],
                                     [0.25, 0.25, 0.5]])    
structure['i_mul_g3c0'] = np.array([[1,   0,    0],
                                     [0.5,  0.5,  0],
                                     [0.5,  0,    0.5],
                                     [0.25, 0.25, 0.5]]) 
###A mask array that relates i_g3_inc to the genotypes that need to be simulated eg:
# 		                g3	
#   g2		BBB	    BBM	BBT	    BMT	
# BBB		TRUE	TRUE	TRUE	TRUE	
# BBM		FALSE	TRUE	FALSE	TRUE	
# BBT		FALSE	FALSE	TRUE	FALSE	
# BMT		FALSE	FALSE	FALSE	TRUE	
  
structure['i_mask_g0g3'] = np.array([[True,True,True,True],
                                     [False,True,False,True],
                                     [False,False,True,True]])    
structure['i_mask_g1g3'] = np.array([[True,True,True,True],
                                     [False,True,False,True],  
                                     [False,False,True,False],  
                                     [False,False,False,True]])   
structure['i_mask_g2g3'] = np.array([[True,True,True,True],
                                     [False,True,False,True],
                                     [False,False,True,False],
                                     [False,False,False,True]])    
structure['i_mask_g3g3'] = np.array([[True,True,True,True],
                                    [False,True,False,True],
                                    [False,False,True,False],
                                    [False,False,False,True]])  
##variations between initial patterns
###lw
structure['i_adjp_lw_initial_w0'] = np.array([0])        
structure['i_adjp_lw_initial_w1'] = np.array([0.0, 0.0,	0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15])
structure['i_adjp_lw_initial_w3'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15])
###cfw
structure['i_adjp_cfw_initial_w0'] = np.array([0])        
structure['i_adjp_cfw_initial_w1'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10])
structure['i_adjp_cfw_initial_w3'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10])
###fd
structure['i_adjp_fd_initial_w0'] = np.array([0])        
structure['i_adjp_fd_initial_w1'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5])
structure['i_adjp_fd_initial_w3'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5])
###fl
structure['i_adjp_fl_initial_w0'] = np.array([0])        
structure['i_adjp_fl_initial_w1'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10])
structure['i_adjp_fl_initial_w3'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10])

##association between management and feedsupply
structure['i_len_m'] = 2
structure['i_len_l'] = 4
structure['i_len_s'] = 5

structure['ia_k2_mlsb1'] =np.array([
                                    [0, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0]
                                    ,[0, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0]
                                    ,[0, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0]
                                    ,[0, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0]
                                    ,[0, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0]
                                    ,[0, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0]
                                    ,[0, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0]
                                    ,[0, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0]
                                    ,[0, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0]
                                    ,[0, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0]
                                    ,[0, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0]
                                    ,[0, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0]
                                    ,[0, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0]
                                    ,[0, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0]
                                    ,[0, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0]
                                    ,[0, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0]
                                    ,[0, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0]
                                    ,[0, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0]
                                    ,[0, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0]
                                    ,[0, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0]
                                    ,[0, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0]
                                    ,[0, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0]
                                    ,[0, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0]
                                    ,[0, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0]
                                    ,[0, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0]
                                    ,[0, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0]
                                    ,[1, 1,	2,	2,	2,	2,	2,	2,	2,	2,	2]
                                    ,[1, 1,	3,	4,	4,	4,	4,	4,	3,	4,	4]
                                    ,[1, 1,	3,	5,	6,	5,	6,	6,	3,	5,	6]
                                    ,[1, 1,	3,	5,	6,	5,	6,	6,	3,	5,	6]
                                    ,[0, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0]
                                    ,[1, 1,	2,	2,	2,	2,	2,	2,	1,	1,	1]
                                    ,[1, 1,	3,	4,	4,	4,	4,	4,	1,	1,	1]
                                    ,[1, 1,	3,	5,	6,	5,	6,	6,	1,	1,	1]
                                    ,[1, 1,	3,	5,	6,	5,	6,	6,	1,	1,	1]
                                    ,[0, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0]
                                    ,[1, 1,	2,	2,	2,	2,	2,	2,	1,	1,	1]
                                    ,[1, 1,	3,	4,	4,	3,	4,	3,	1,	1,	1]
                                    ,[1, 1,	3,	5,	6,	3,	5,	3,	1,	1,	1]
                                    ,[1, 1,	3,	5,	6,	3,	5,	3,	1,	1,	1]])



##association between management and postprocessing clustering
structure['i_n_v1type'] = 3
structure['i_k2_idx_dams'] = np.array([
['NM',   '00',   '11',	 '22',   '33',   '21',   '32',   '31',   '10',   '20',   '30'],
['NM-1', '00-1', '11-1', '22-1', '33-1', '21-1', '32-1', '31-1', '10-1', '20-1', '30-1'],
['NM-2', '00-2', '11-2', '22-2', '33-2', '21-2', '32-2', '31-2', '10-2', '20-2', '30-2']])

structure['ia_ppk2g1_vlsb1'] = np.array([
                                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                                        , [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                                        , [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                                        , [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                                        , [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                                        , [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                                        , [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                                        , [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                                        , [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                                        , [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                                        , [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                                        , [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                                        , [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                                        , [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                                        , [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                                        , [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                                        , [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                                        , [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                                        , [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                                        , [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                                        , [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                                        , [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2]
                                        , [0, 1, 2, 3, 3, 3, 3, 3, 2, 3, 3]
                                        , [0, 1, 2, 3, 4, 3, 4, 4, 2, 3, 4]
                                        , [0, 1, 2, 3, 4, 3, 4, 4, 2, 3, 4]
                                        , [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                                        , [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2]
                                        , [0, 1, 2, 3, 3, 3, 3, 3, 2, 3, 3]
                                        , [0, 1, 2, 3, 4, 3, 4, 4, 2, 3, 4]
                                        , [0, 1, 2, 3, 4, 3, 4, 4, 2, 3, 4]
                                        , [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                                        , [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2]
                                        , [0, 1, 2, 3, 3, 3, 3, 3, 2, 3, 3]
                                        , [0, 1, 2, 3, 4, 3, 4, 4, 2, 3, 4]
                                        , [0, 1, 2, 3, 4, 3, 4, 4, 2, 3, 4]
                                        , [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                                        , [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2]
                                        , [0, 1, 2, 3, 3, 3, 3, 3, 2, 3, 3]
                                        , [0, 1, 2, 3, 4, 3, 4, 4, 2, 3, 4]
                                        , [0, 1, 2, 3, 4, 3, 4, 4, 2, 3, 4]
                                        , [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                                        , [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                                        , [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                                        , [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                                        , [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                                        , [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                                        , [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2]
                                        , [0, 1, 2, 3, 3, 3, 3, 3, 2, 3, 3]
                                        , [0, 1, 2, 3, 4, 3, 4, 4, 2, 3, 4]
                                        , [0, 1, 2, 3, 4, 3, 4, 4, 2, 3, 4]
                                        , [0, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1]
                                        , [0, 1, 2, 2, 2, 2, 2, 2, 8, 8, 8]
                                        , [0, 1, 2, 3, 3, 3, 3, 3, 8, 9, 9]
                                        , [0, 1, 2, 3, 4, 3, 4, 4, 8, 9, 10]
                                        , [0, 1, 2, 3, 4, 3, 4, 4, 8, 9, 10]
                                        , [0, 1, 2, 3, 4, 2, 3, 2, 1, 1, 1]
                                        , [0, 1, 2, 3, 4, 2, 3, 2, 8, 8, 8]
                                        , [0, 1, 2, 3, 4, 5, 3, 5, 8, 9, 9]
                                        , [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                                        , [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
##association between management and postprocessing clustering - offs
structure['i_k5_idx_offs'] = np.array([['11',	'22',	'33',	'21',	'32',	'31'],
                                       ['00-1',	'11-1',	'22-1',	'33-1',	'21-1',	'32-1'],
                                       ['00-2',	'11-2',	'22-2',	'33-2',	'21-2',	'32-2']])

structure['ia_ppk5_lsb0'] = np.array([
                                        [0,	0,	0,	0,	0,	0],
                                        [0,	0,	0,	0,	0,	0],
                                        [0,	0,	0,	0,	0,	0],
                                        [0,	0,	0,	0,	0,	0],
                                        [0,	0,	0,	0,	0,	0],
                                        [0,	0,	0,	0,	0,	0],
                                        [0,	0,	0,	0,	0,	0],
                                        [0,	1,	1,	1,	1,	1],
                                        [0,	1,	2,	1,	2,	2],
                                        [0,	1,	2,	1,	2,	2],
                                        [0,	0,	0,	0,	0,	0],
                                        [0,	0,	0,	0,	0,	0],
                                        [0,	1,	1,	1,	1,	1],
                                        [0,	1,	2,	1,	2,	2],
                                        [0,	1,	2,	1,	2,	2],
                                        [0,	1,	2,	0,	1,	0],
                                        [0,	1,	2,	0,	1,	0],
                                        [0,	1,	2,	3,	1,	3],
                                        [0,	1,	2,	3,	4,	5],
                                        [0,	1,	2,	3,	4,	5]])



########################
#period                #
########################
##Length of standard labour period, must be an integer that 12 is divisible by
structure['labour_period_len'] = relativedelta(months=1)


##############
#phases      #
##############
##the number of previous land uses considered for crop inputs - when this changes yeild input and fert and chem will need to be expended to include the extra years previous land use
structure['num_prev_phase']=1

#number of phases analysed ie rotation length if you will (although not really a rotation)
structure['phase_len'] = 6

#rotation phases and constraints read in from excel 
structure['phases'] = pd.read_excel('Rotation.xlsx', sheet_name='rotation list', header= None, index_col = 0).T.reset_index(drop=True).T  #reset the col headers to std ie 0,1,2 etc



###############
#landuses     #
###############
'''
A1, E1 are special sets used in con2 - currently not used
Note
- A1 is also used in pasture functions to build the germ df, so it cant be deleted
- C is used in stubble module, createmodel & mach
- C1 is used just in pasture functions
- sets now include capitals - this shouldnt effect con1 but it makes building the germ df easier
'''
##special sets that are used elsewhere from rotations
###used to make nap inputs - note cont lucerne and tedera are added seperately at the end of the cost section hence not inlcuded here.
# structure['All_pas']={'a', 'ar', 'a3', 'a4', 'a5'
#                 , 's', 'sr', 's3', 's4', 's5'
#                 , 'm', 'm3', 'm4', 'm5'
#                 , 'u', 'ur', 'u3', 'u4', 'u5'
#                 , 'x', 'xr', 'x3', 'x4', 'x5'
#                 , 'j', 't', 'jr', 'tr'
#                 }
##all pas2 includes cont pasture - used in reporting
structure['All_pas']={'a', 'ar'
                , 's', 'sr'
                , 'm'
                , 'u', 'ur','uc'
                , 'x', 'xr','xc'
                , 'j', 't', 'jr', 'tr','tc','jc'
                }
##next set is used in pasture.py for germination and phase area
structure['pasture_sets']={'annual': {'a', 'ar'
                                , 's', 'sr'
                                , 'm'}
                        ,'lucerne':{'u', 'uc', 'ur'
                                   , 'x', 'xc', 'xr'}
                        ,'tedera':{'j','jc', 't','tc', 'jr', 'tr'}
                       }
##G and C1 are just used in pas.py for germination ^can be removed when germination is calculated from sim
structure['G']={'b', 'h', 'o','of', 'w', 'f','i', 'k', 'l', 'v', 'z','r'
                , 'a', 'ar'
                , 's', 'sr'
                , 'm'
                , 'u', 'ur'
                , 'x', 'xr'
                , 'j', 't', 'jr', 'tr'
                , 'G', 'Y', 'E', 'N', 'P', 'OF'
                , 'A', 'AR'
                , 'S', 'SR'
                , 'M'
                , 'U'
                , 'X'
                , 'T', 'J'} #all landuses
structure['C1']={'E', 'N', 'P', 'OF', 'b', 'h', 'o', 'of', 'w', 'f','i', 'k', 'l', 'v', 'z','r'} #had to create a seperate set because don't want the capitatl in the crop set above as it is used to create pyomo set 


structure['All']={'b', 'h', 'o', 'of', 'w', 'f','i', 'k', 'l', 'v', 'z','r', 'a', 'ar', 's', 'sr', 'm', 'u', 'uc', 'ur', 'x', 'xc', 'xr', 'j','jc', 't','tc', 'jr', 'tr'} #used in reporting and bounds
structure['C']={'b', 'h', 'o', 'of', 'w', 'f','i', 'k', 'l', 'v', 'z','r'} #all crops, used in stubble and mach (not used for rotations)
structure['Hay']={'h'} #all crops that produce hay - used in machpyomo/coremodel for hay con
##special sets used in crop sim
structure['Ys'] = {'Y'}
structure['As'] = {'A','a'}
structure['JR'] = {'jr'}
structure['TR'] = {'tr'}
structure['UR'] = {'ur'}
structure['XR'] = {'xr'}
structure['PAS'] = {'A', 'AR', 'S', 'SR', 'M','T','J','U','X', 'tc', 'jc', 'uc', 'xc'} 
##sets used in to build rotations
structure['A']={'a', 'ar','s', 'sr', 'm'
                , 'A', 'AR'
                , 'S', 'SR'
                , 'M'} #annual
structure['A1']={'a',  's', 'm'} #annual not resown - special set used in pasture germ and con2 when determining if a rotatin provides a rotation because in yr1 we dont want ar to provide an A bevause we need to distinguish beteween them
structure['AR']={'ar', 'AR'} #resown annual
structure['E']={'E', 'E1', 'OF', 'b', 'h', 'o', 'of', 'w'} #cereals
structure['E1']={'E', 'b', 'h', 'o', 'w'} #cereals
# # structure['H']={'h', 'of'} #non harvested cereals
structure['J']={'J', 'j', 'jr'} #tedera
structure['M']={'m', 'M'} #manipulated pasture
structure['N']={'N', 'z','r'} #canolas
structure['OF']={'OF', 'of'} #oats fodder
structure['P']={'P', 'f','i', 'k', 'l', 'v'} #pulses
structure['S']={'s','sr', 'S', 'SR'} #spray topped pasture
structure['SR']={'sr', 'SR'} #spray topped pasture
structure['T']={'T', 't', 'tr','J', 'j', 'jr'} #tedera - also includes manipulated tedera because it is combined in yrs 3,4,5
structure['U']={'u', 'ur', 'U','x', 'xr', 'X'} #lucerne
structure['X']={'x', 'xr', 'X'} #lucerne
structure['Y']={'b', 'h', 'o','of', 'w', 'f','i', 'k', 'l', 'v', 'z','r'
                , 'Y', 'E', 'E1', 'N', 'P', 'OF'} #anything not pasture


'''make each landuse a set so the issuperset func works'''
structure['a']={'a'}
structure['ar']={'ar'}
structure['b']={'b'}
structure['f']={'f'}
structure['h']={'h'}
structure['i']={'i'}
structure['j']={'j'}
structure['jc']={'jc'}
structure['jr']={'jr'}
structure['k']={'k'}
structure['l']={'l'}
structure['m']={'m'}
structure['o']={'o'}
structure['of']={'of'}
structure['r']={'r'}
structure['s']={'s'}
structure['sr']={'sr'}
structure['t']={'t'}
structure['tc']={'tc'}
structure['tr']={'tr'}
structure['u']={'u'}
structure['uc']={'uc'}
structure['ur']={'ur'}
structure['v']={'v'}
structure['w']={'w'}
structure['x']={'x'}
structure['xc']={'xc'}
structure['xr']={'xr'}
structure['z']={'z'}











# phases = phases[~(np.isin(phases[:,i], ['U'])&np.isin(phases[:,i+1], ['U4','U5','u4','u5']))] #only U or U3 ufter U
#     phases = phases[~(np.isin(phases[:,i], ['U3'])&np.isin(phases[:,i+1], ['U', 'U3','U5','u','ur','u3','u5']))] #pasture 4 muxt come ufter pasture 3
#     phases = phases[~(np.isin(phases[:,i], ['U4'])&np.isin(phases[:,i+1], ['U', 'U3','U4','u','ur','u3','u4']))] #pasture 5 muxt come ufter pasture 4
#     phases = phases[~(np.isin(phases[:,i], ['U5'])&np.isin(phases[:,i+1], ['U', 'U3','U4','u','ur','u3','u4']))] #pasture 5 muxt come ufter pasture 5
#     phases = phases[~(~np.isin(phases[:,i], ['U'])&np.isin(phases[:,i+1], ['U3','u3']))] #cant have U3 after anything except U
#     try:  #used for conditions that are concerned with more than two yrs
#         phases = phases[~(~np.isin(phases[:,i], ['U'])&np.isin(phases[:,i+2], ['U3','u3']))] #cant have U3 ufter unything except U U (this is the second part to the rule above)
#     except IndexError: pass
#     phases = phases[~(~np.isin(phases[:,i], ['U3'])&np.isin(phases[:,i+1], ['U4','u4']))] #cant have U4 after anything except U3
#     phases = phases[~(~np.isin(phases[:,i], ['U4'])&np.isin(phases[:,i+1], ['U5','u5']))] #cant have U5 after anything except U4
#     try:  #used for conditions that are concerned with more than two yrs
#         phases = phases[~(np.isin(phases[:,i], ['U'])&np.isin(phases[:,i+1], ['U'])&~np.isin(phases[:,i+2], ['U3','u3']))] #can only huve U3 ufter U U (huve uxed u double negitive here)
#     except IndexError: pass

#     ##Manipulated Lucerne
#     phases = phases[~(np.isin(phases[:,i], ['X'])&np.isin(phases[:,i+1], ['X4','X5','x4','x5']))] #only U or U3 ufter U
#     phases = phases[~(np.isin(phases[:,i], ['X3'])&np.isin(phases[:,i+1], ['X','X3','X5','x','xr','x3','x5']))] #pasture 4 muxt come ufter pasture 3
#     phases = phases[~(np.isin(phases[:,i], ['X4'])&np.isin(phases[:,i+1], ['X','X3','X4','x','xr','x3','x4']))] #pasture 5 muxt come ufter pasture 4
#     phases = phases[~(np.isin(phases[:,i], ['X5'])&np.isin(phases[:,i+1], ['X','X3','X4','x','xr','x3','x4']))] #pasture 5 muxt come ufter pasture 5
#     phases = phases[~(~np.isin(phases[:,i], ['X'])&np.isin(phases[:,i+1], ['X3','x3']))] #cant have U3 after anything except U
#     try:  #used for conditions that are concerned with more than two yrs
#         phases = phases[~(~np.isin(phases[:,i], ['X'])&np.isin(phases[:,i+2], ['X3','x3']))] #cant have U3 ufter unything except U U (this is the second part to the rule above)
#     except IndexError: pass
#     phases = phases[~(~np.isin(phases[:,i], ['X3'])&np.isin(phases[:,i+1], ['X4','x4']))] #cant have U4 after anything except U3
#     phases = phases[~(~np.isin(phases[:,i], ['X4'])&np.isin(phases[:,i+1], ['X5','x5']))] #cant have U5 after anything except U4
#     try:  #used for conditions that are concerned with more than two yrs
#         phases = phases[~(np.isin(phases[:,i], ['X'])&np.isin(phases[:,i+1], ['X'])&~np.isin(phases[:,i+2], ['X3','x3']))] #can only huve U3 ufter U U (huve uxed u double negitive here)
#     except IndexError: pass

# #Lucerne
#
#########################################################################################################################################################################################################
#########################################################################################################################################################################################################
#universal functions that use data from above
#########################################################################################################################################################################################################
#########################################################################################################################################################################################################


#Function that just uses inout inputs but is used in multiple other pre-calc modules
#defined here to limit imorting pre calc modules in other precalc modules
def cols():
    #this is used to make a list of the relevent column numbers used in merge function, to specify the columns that are being matched - it will change if inputs specifying number of phases changes
    cols = []
    for i in reversed(range(structure['num_prev_phase']+1)):
        cols.append(structure['phase_len']-1-i) 
    return cols


