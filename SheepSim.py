# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 09:35:18 2020

@author: John
"""

'''
import functions from other modules
'''
# import datetime as dt
import pandas as pd
import numpy as np

# from numba import jit

# import FeedBudget as fdb
# import Functions as fun
# import Periods as per
import PropertyInputs as pinp
import SheepSimRoutines as sfun
import UniversalInputs as uinp


############################
### _constants required    #
############################
## define some parameters required to size arrays.
# n_feed_pools        = uinp.n_feed_pools
# n_feed_periods      = len(pinp.feed_inputs['feed_periods']) - 1

#^ put the values as lists in universal.xlsx (SheepDefinitions!)
# then define n by length of the list
n_animal_types = 2      # a: wool, meat
n_btrt = 6              # b: 11, 22, 21, 33, 32, 31
n_genotype_options = 5  # c: number in spreadhseet
n_dam_ages = 3          # d: yearling, maiden, adult
n_max_ecycles = 2       # e: max number of estrus cycles they are joined
n_feed_var_periods = 10 # f:
n_genotypes = 6         # g: B, M, T, BM, BT, BMT
n_g0 = 3                # g0: B, M, T
n_g1 = 2                # g1: BB, BM
n_g2 = 4                # g2: BBB, BBM, BBT, BMT
n_husbandry_class = 1   # h
n_groups_rams = 3       # i & g0: genotypes of rams
n_groups_ewes = 4       # j: genotype groups of ewes
n_groups_offspring = 5  # k: genotype groups and growth profile
n_groups_lambing = 1    # l: lambing groups for the seed animals (1 unless doing a TOL analysis or 8 month joinings)
n_months = 12           # m: Jan to Dec
n_feed_periods = 10     # n:
n_lambing_opps = 15     # o:
# n_sim_periods         # p: see below
n_labour_periods = 16   # q:
n_feed_variables        # r:
n_shearing_occs = 16    # s:
n_sale_times = 4        # t: weaner, backgrounded, finished, remainder
n_husbandry_options = 10# u:
n_genders = 3           # w: ewe, wether, ram
n_litter_size = 5       # x: Dry, single, twin, triplet, not mated
n_lactation_number = 5  # y: dry, single, twin, triplet, in utero
n_sexes = 3             # w: ram, ewe, wether
# n_sim_periods see below
n_labour_periods = 16   # q
        i_sim_periods_year = 52 # ^uinp.n_sim_periods_year   will be in structure dict now
i_oldest_animal= 6.5    # ^uinp.i_oldest_animal

birth_date_i = uinp.propertydata['ExcelName']   #Find the ExcelNames
birth_date_jl = uinp.propertydata['ExcelName']
birth_date_jel = uinp.propertydata['ExcelName']
birth_date_kel = uinp.propertydata['ExcelName']
## Some one time data manipulation for the inputs just read
start_year = np.min(birth_date_jl)
## might need to test and rebase the year for the other animal groups


### _define the periods
n_sim_periods, date_p, p_index_p, step \
        = sfun.sim_periods(start_year, i_sim_periods_year, i_oldest_animal)
### _array dimensions
va      = (n_genders
          ,n_animal_types
          )
xa      = (n_litter_size
          ,n_animal_types
          )
ya      = (n_lactation_number
          ,n_animal_types
          )
vg      = (n_genders
          ,n_genotypes
          )
xg      = (n_litter_size
          ,n_genotypes
          )
yg      = (n_lactation_number
          ,n_genotypes
          )
##          ,23    #  this dimension represents the subscript from GrazPLan
                  #  not sure how these values vary with genotype and lactation number
                  #  so not sure how to pass this to sfun.intake
## the number required varies with the constatnt= being defined and therefore
## the value will be specified during the instantiation of the numpy array
          )
il      = (n_groups_rams
          ,n_groups_lambing
          )
d_is    = (n_groups_rams
          ,n_shearing_occs
          )
jel     = (n_groups_ewes
          ,n_max_ecycles
          ,n_groups_lambing
          )
jewbl   = (n_groups_ewes
          ,n_max_ecycles
          ,n_sexes
          ,n_btrt
          ,n_groups_lambing
          )
jexl    = (n_groups_ewes
          ,n_max_ecycles
          ,n_litter_size
          ,n_groups_lambing
          )
jexyl   = (n_groups_ewes
          ,n_max_ecycles
          ,n_litter_size
          ,n_lactation_number
          ,n_groups_lambing
          )
jl      = (n_groups_ewes
          ,n_groups_lambing
          )
jo      = (n_groups_ewes
          ,n_lambing_opps
          )
js      = (n_groups_ewes
          ,n_shearing_occs
          )
jwexyl  = (n_groups_ewes
          ,n_sexes
          ,n_max_ecycles
          ,n_litter_size
          ,n_lactation_number
          ,n_groups_lambing
          )
jxyl    = (n_groups_ewes
          ,n_litter_size
          ,n_lactation_number
          ,n_groups_lambing
          )
kd      = (n_groups_offspring
          ,n_dam_ages
          )
kdwbl   = (n_groups_offspring
          ,n_dam_ages
          ,n_sexes
          ,n_btrt
          ,n_sexes
          )
kdwebl  = (n_groups_offspring
          ,n_dam_ages
          ,n_sexes
          ,n_max_ecycles
          ,n_btrt
          ,n_groups_lambing
          )
kel     = (n_groups_offspring
          ,n_max_ecycles
          ,n_groups_lambing
          )
ks      = (n_groups_offspring
          ,n_shearing_occs
          )
ojel    = (n_lambing_opps
          ,n_groups_ewes
          ,n_max_ecycles
          ,n_groups_lambing
          )
pi      = (n_sim_periods
          ,n_groups_rams
          )
pil     = (n_sim_periods
          ,n_groups_rams
          ,n_groups_lambing
          )
pjel    = (n_sim_periods
          ,n_groups_ewes
          ,n_max_ecycles
          ,n_groups_lambing
          )
pjl     = (n_sim_periods
          ,n_max_ecycles
          ,n_groups_lambing
          )
pjxyl   = (n_sim_periods
          ,n_max_ecycles
          ,n_litter_size
          ,n_lactation_number
          ,n_groups_lambing
          )
pjl     = (n_sim_periods
          ,n_groups_ewes
          ,n_groups_lambing
          )
pkdwbl  = (n_sim_periods
          ,n_groups_offspring
          ,n_dam_ages
          ,n_sexes
          ,n_btrt
          ,n_groups_lambing
          )
pkdwebl = (n_sim_periods
          ,n_groups_offspring
          ,n_dam_ages
          ,n_sexes
          ,n_max_ecycles
          ,n_btrt
          ,n_groups_lambing
          )
pkdwl   = (n_sim_periods
          ,n_groups_offspring
          ,n_dam_ages
          ,n_sexes
          ,n_groups_lambing
          )
pkel    = (n_sim_periods
          ,n_groups_offspring
          ,n_max_ecycles
          ,n_groups_lambing
          )
pr      = (n_sim_periods
          ,n_feed_variables
          )



###################################
### initialise global arrays      #
###################################
## Instantiate the globals arrays
## # these store the output of simulation and the parameters for pyomo
## # see documentation for a description of each variable
n_i                     = np.zeros(n_groups_rams , dtype = 'float64')
pi_i                    = np.zeros(n_groups_rams , dtype = 'float64')
mei_i                   = np.zeros(n_groups_rams , dtype = 'float64')
lw_ffcf_i               = np.zeros(n_groups_rams , dtype = 'float64')
lw_i                    = np.zeros(n_groups_rams , dtype = 'float64')
aw_i                    = np.zeros(n_groups_rams , dtype = 'float64')
mw_i                    = np.zeros(n_groups_rams , dtype = 'float64')
bw_i                    = np.zeros(n_groups_rams , dtype = 'float64')
ww_i                    = np.zeros(n_groups_rams , dtype = 'float64')
gw_i                    = np.zeros(n_groups_rams , dtype = 'float64')
ss_i                    = np.zeros(n_groups_rams , dtype = 'float64')
wool_value_i            = np.zeros(n_groups_rams , dtype = 'float64')
sale_value_i            = np.zeros(n_groups_rams , dtype = 'float64')
ch4_nggi_i              = np.zeros(n_groups_rams , dtype = 'float64')
n20_nggi_i              = np.zeros(n_groups_rams , dtype = 'float64')
ch4_bc_i                = np.zeros(n_groups_rams , dtype = 'float64')

n_rams_il               = np.zeros(il , dtype = 'float64')
n_jexyl                 = np.zeros(jexyl , dtype = 'float64')
pi_jexyl                = np.zeros(jexyl , dtype = 'float64')
mei_jexyl               = np.zeros(jexyl , dtype = 'float64')
lw_ffcf_jexyl           = np.zeros(jexyl , dtype = 'float64')
lw_jexyl                = np.zeros(jexyl , dtype = 'float64')
aw_jexyl                = np.zeros(jexyl , dtype = 'float64')
mw_jexyl                = np.zeros(jexyl , dtype = 'float64')
bw_jexyl                = np.zeros(jexyl , dtype = 'float64')
ww_jexyl                = np.zeros(jexyl , dtype = 'float64')
gw_jexyl                = np.zeros(jexyl , dtype = 'float64')
ss_jexyl                = np.zeros(jexyl , dtype = 'float64')
wool_value_jexyl        = np.zeros(jexyl , dtype = 'float64')
sale_value_jexyl        = np.zeros(jexyl , dtype = 'float64')
ch4_nggi_jexyl          = np.zeros(jexyl , dtype = 'float64')
n20_nggi_jexyl          = np.zeros(jexyl , dtype = 'float64')
ch4_bc_jexyl            = np.zeros(jexyl , dtype = 'float64')
n_jwexyl                = np.zeros(jwexyl , dtype = 'float64')
pi_jwexyl               = np.zeros(jwexyl , dtype = 'float64')
mei_jwexyl              = np.zeros(jwexyl , dtype = 'float64')
lw_ffcf_jwexyl          = np.zeros(jwexyl , dtype = 'float64')
lw_jwexyl               = np.zeros(jwexyl , dtype = 'float64')
aw_jwexyl               = np.zeros(jwexyl , dtype = 'float64')
mw_jwexyl               = np.zeros(jwexyl , dtype = 'float64')
bw_jwexyl               = np.zeros(jwexyl , dtype = 'float64')
ww_jwexyl               = np.zeros(jwexyl , dtype = 'float64')
gw_jwexyl               = np.zeros(jwexyl , dtype = 'float64')
ss_jwexyl               = np.zeros(jwexyl , dtype = 'float64')
wool_value_jwexyl       = np.zeros(jwexyl , dtype = 'float64')
sale_value_jwexyl       = np.zeros(jwexyl , dtype = 'float64')
ch4_nggi_jwexyl         = np.zeros(jwexyl , dtype = 'float64')
n20_nggi_jwexyl         = np.zeros(jwexyl , dtype = 'float64')
ch4_bc_jwexyl           = np.zeros(jwexyl , dtype = 'float64')
n_kdwebl                = np.zeros(kdwebl , dtype = 'float64')
pi_kdwebl               = np.zeros(kdwebl , dtype = 'float64')
mei_kdwebl              = np.zeros(kdwebl , dtype = 'float64')
lw_ffcf_kdwebl          = np.zeros(kdwebl , dtype = 'float64')
lw_kdwebl               = np.zeros(kdwebl , dtype = 'float64')
aw_kdwebl               = np.zeros(kdwebl , dtype = 'float64')
mw_kdwebl               = np.zeros(kdwebl , dtype = 'float64')
bw_kdwebl               = np.zeros(kdwebl , dtype = 'float64')
ww_kdwebl               = np.zeros(kdwebl , dtype = 'float64')
gw_kdwebl               = np.zeros(kdwebl , dtype = 'float64')
ss_kdwebl               = np.zeros(kdwebl , dtype = 'float64')
wool_value_kdwebl       = np.zeros(kdwebl , dtype = 'float64')
sale_value_kdwebl       = np.zeros(kdwebl , dtype = 'float64')
ch4_nggi_kdwebl         = np.zeros(kdwebl , dtype = 'float64')
n20_nggi_kdwebl         = np.zeros(kdwebl , dtype = 'float64')
ch4_bc_kdwebl           = np.zeros(kdwebl , dtype = 'float64')
p_n_pi                  = np.zeros(pi , dtype = 'float64')
p_pi_pi                 = np.zeros(pi , dtype = 'float64')
p_mei_pi                = np.zeros(pi , dtype = 'float64')
r_lw_ffcf_pi            = np.zeros(pi , dtype = 'float64')
r_lw_pi                 = np.zeros(pi , dtype = 'float64')
r_aw_pi                 = np.zeros(pi , dtype = 'float64')
r_mw_pi                 = np.zeros(pi , dtype = 'float64')
r_bw_pi                 = np.zeros(pi , dtype = 'float64')
r_ww_pi                 = np.zeros(pi , dtype = 'float64')
r_gw_pi                 = np.zeros(pi , dtype = 'float64')
p_ss_pi                 = np.zeros(pi , dtype = 'float64')
p_wool_value_pi         = np.zeros(pi , dtype = 'float64')
p_sale_value_pi         = np.zeros(pi , dtype = 'float64')
p_ch4_nggi_pi           = np.zeros(pi , dtype = 'float64')
p_n20_nggi_pi           = np.zeros(pi , dtype = 'float64')
p_ch4_bc_pi             = np.zeros(pi , dtype = 'float64')
p_n_pjexyl              = np.zeros(pjexyl , dtype = 'float64')
p_pi_pjexyl             = np.zeros(pjexyl , dtype = 'float64')
p_mei_pjexyl            = np.zeros(pjexyl , dtype = 'float64')
r_lw_ffcf_pjexyl        = np.zeros(pjexyl , dtype = 'float64')
r_lw_pjexyl             = np.zeros(pjexyl , dtype = 'float64')
r_aw_pjexyl             = np.zeros(pjexyl , dtype = 'float64')
r_mw_pjexyl             = np.zeros(pjexyl , dtype = 'float64')
r_bw_pjexyl             = np.zeros(pjexyl , dtype = 'float64')
r_ww_pjexyl             = np.zeros(pjexyl , dtype = 'float64')
r_gw_pjexyl             = np.zeros(pjexyl , dtype = 'float64')
p_ss_pjexyl             = np.zeros(pjexyl , dtype = 'float64')
p_wool_value_pjexyl     = np.zeros(pjexyl , dtype = 'float64')
p_sale_value_pjexyl     = np.zeros(pjexyl , dtype = 'float64')
p_ch4_nggi_pjexyl       = np.zeros(pjexyl , dtype = 'float64')
p_n20_nggi_pjexyl       = np.zeros(pjexyl , dtype = 'float64')
p_ch4_bc_pjexyl         = np.zeros(pjexyl , dtype = 'float64')
p_n_pjwexyl             = np.zeros(pjwexyl , dtype = 'float64')
p_pi_pjwexyl            = np.zeros(pjwexyl , dtype = 'float64')
p_mei_pjwexyl           = np.zeros(pjwexyl , dtype = 'float64')
r_lw_ffcf_pjwexyl       = np.zeros(pjwexyl , dtype = 'float64')
r_lw_pjwexyl            = np.zeros(pjwexyl , dtype = 'float64')
r_aw_pjwexyl            = np.zeros(pjwexyl , dtype = 'float64')
r_mw_pjwexyl            = np.zeros(pjwexyl , dtype = 'float64')
r_bw_pjwexyl            = np.zeros(pjwexyl , dtype = 'float64')
r_ww_pjwexyl            = np.zeros(pjwexyl , dtype = 'float64')
r_gw_pjwexyl            = np.zeros(pjwexyl , dtype = 'float64')
p_ss_pjwexyl            = np.zeros(pjwexyl , dtype = 'float64')
p_wool_value_pjwexyl    = np.zeros(pjwexyl , dtype = 'float64')
p_sale_value_pjwexyl    = np.zeros(pjwexyl , dtype = 'float64')
p_ch4_nggi_pjwexyl      = np.zeros(pjwexyl , dtype = 'float64')
p_n20_nggi_pjwexyl      = np.zeros(pjwexyl , dtype = 'float64')
p_ch4_bc_pjwexyl        = np.zeros(pjwexyl , dtype = 'float64')
p_n_pkdwebl             = np.zeros(pkdwebl , dtype = 'float64')
p_pi_pkdwebl            = np.zeros(pkdwebl , dtype = 'float64')
p_mei_pkdwebl           = np.zeros(pkdwebl , dtype = 'float64')
r_lw_ffcf_pkdwebl       = np.zeros(pkdwebl , dtype = 'float64')
r_lw_pkdwebl            = np.zeros(pkdwebl , dtype = 'float64')
r_aw_pkdwebl            = np.zeros(pkdwebl , dtype = 'float64')
r_mw_pkdwebl            = np.zeros(pkdwebl , dtype = 'float64')
r_bw_pkdwebl            = np.zeros(pkdwebl , dtype = 'float64')
r_ww_pkdwebl            = np.zeros(pkdwebl , dtype = 'float64')
r_gw_pkdwebl            = np.zeros(pkdwebl , dtype = 'float64')
p_ss_pkdwebl            = np.zeros(pkdwebl , dtype = 'float64')
p_wool_value_pkdwebl    = np.zeros(pkdwebl , dtype = 'float64')
p_sale_value_pkdwebl    = np.zeros(pkdwebl , dtype = 'float64')
p_ch4_nggi_pkdwebl      = np.zeros(pkdwebl , dtype = 'float64')
p_n20_nggi_pkdwebl      = np.zeros(pkdwebl , dtype = 'float64')
p_ch4_bc_pkdwebl        = np.zeros(pkdwebl , dtype = 'float64')


def simulation():
    """
    A function to wrap the simulation that can be called by SheepPyomo.

    Called after the sensitivty variables have been updated.
    It populates the arrays by looping through the time periods
    Globally define arrays are used to transfer results to sheep_paramters()

    Returns
    -------
    None.
    """
    ############################
    ### initialise arrays      #
    ############################
    ## Instantiate the arrays that are only required within this function
    ## mainly arrays that will store the input data that require pre-defining
    ## # see documentation for a description of each variable
    a_c_g0              = np.zeros(g0, dtype = 'float64')
    a_maternal_g0_g1    = np.zeros(g1, dtype = 'float64')
    a_paternal_g0_g1    = np.zeros(g1, dtype = 'float64')
    a_maternal_g1_g2    = np.zeros(g2, dtype = 'float64')
    a_paternal_g0_g2    = np.zeros(g2, dtype = 'float64')

    c_cn_vg             = np.zeros(7, vg, dtype = 'float64')
    c_cr_g              = np.zeros(23, n_genotypes, dtype = 'float64')
    c_ck_g              = np.zeros(18, n_genotypes, dtype = 'float64')
    c_cm_vg             = np.zeros(20, vg, dtype = 'float64')
    c_cw_g              = np.zeros(15, n_genotypes, dtype = 'float64')
    c_cc_g              = np.zeros(17, n_genotypes, dtype = 'float64')
    c_cg_g              = np.zeros(19, n_genotypes, dtype = 'float64')
    c_ch_g              = np.zeros(n_genotypes, dtype = 'float64')
    c_cd_g              = np.zeros(n_genotypes, dtype = 'float64')
    c_sfw_g             = np.zeros(n_genotypes, dtype = 'float64')
    a_g_j               = np.zeros(n_groups_ewes , dtype = 'float64')
    a_w_j               = np.zeros(n_groups_ewes , dtype = 'float64')
    a_g_i               = np.zeros(n_groups_rams , dtype = 'float64')
    a_w_i               = np.zeros(n_groups_rams , dtype = 'float64')
    mr_cs_i             = np.zeros(n_groups_rams , dtype = 'float64')
    mr_mu_i             = np.zeros(n_groups_rams , dtype = 'float64')
    nw_i                = np.zeros(n_groups_rams , dtype = 'float64')
    lw_ffcf_start_i     = np.zeros(n_groups_rams , dtype = 'float64')
    aw_start_i          = np.zeros(n_groups_rams , dtype = 'float64')
    mw_start_i          = np.zeros(n_groups_rams , dtype = 'float64')
    bw_start_i          = np.zeros(n_groups_rams , dtype = 'float64')
    relsize_i           = np.zeros(n_groups_rams , dtype = 'float64')
    relsize1_i          = np.zeros(n_groups_rams , dtype = 'float64')
    zf1_i               = np.zeros(n_groups_rams , dtype = 'float64')
    zf2_i               = np.zeros(n_groups_rams , dtype = 'float64')
    rc_i                = np.zeros(n_groups_rams , dtype = 'float64')
    cfw_start_i         = np.zeros(n_groups_rams , dtype = 'float64')
    fl_start_i          = np.zeros(n_groups_rams , dtype = 'float64')
    fd_start_i          = np.zeros(n_groups_rams , dtype = 'float64')
    fd_min_start_i      = np.zeros(n_groups_rams , dtype = 'float64')
    foo_i               = np.zeros(n_groups_rams , dtype = 'float64')
    dmd_i               = np.zeros(n_groups_rams , dtype = 'float64')
    md_i                = np.zeros(n_groups_rams , dtype = 'float64')
    hf_i                = np.zeros(n_groups_rams , dtype = 'float64')
    meme_i              = np.zeros(n_groups_rams , dtype = 'float64')
    level_i             = np.zeros(n_groups_rams , dtype = 'float64')
    d_lw_f_i            = np.zeros(n_groups_rams , dtype = 'float64')
    cw_i                = np.zeros(n_groups_rams , dtype = 'float64')
    mec_i               = np.zeros(n_groups_rams , dtype = 'float64')
    ldr_i               = np.zeros(n_groups_rams , dtype = 'float64')
    lb_i                = np.zeros(n_groups_rams , dtype = 'float64')
    mel_i               = np.zeros(n_groups_rams , dtype = 'float64')
    d_cfw_wolag_i       = np.zeros(n_groups_rams , dtype = 'float64')
    d_cfw_i             = np.zeros(n_groups_rams , dtype = 'float64')
    mew_i               = np.zeros(n_groups_rams , dtype = 'float64')
    d_fd_i              = np.zeros(n_groups_rams , dtype = 'float64')
    d_fl_i              = np.zeros(n_groups_rams , dtype = 'float64')
    mecold_i            = np.zeros(n_groups_rams , dtype = 'float64')
    kg_i                = np.zeros(n_groups_rams , dtype = 'float64')
    ebg_i               = np.zeros(n_groups_rams , dtype = 'float64')
    pg_i                = np.zeros(n_groups_rams , dtype = 'float64')
    lw_ffcf_max_i       = np.zeros(n_groups_rams , dtype = 'float64')
    fw_end_i            = np.zeros(n_groups_rams , dtype = 'float64')
    cfw_i               = np.zeros(n_groups_rams , dtype = 'float64')
    fl_i                = np.zeros(n_groups_rams , dtype = 'float64')
    fd_min_i            = np.zeros(n_groups_rams , dtype = 'float64')
    fd_i                = np.zeros(n_groups_rams , dtype = 'float64')
    ldr_end_i           = np.zeros(n_groups_rams , dtype = 'float64')
    lb_end_i            = np.zeros(n_groups_rams , dtype = 'float64')
    date_p              = np.zeros(n_sim_periods , dtype = 'float64')
    doy_p               = np.zeros(n_sim_periods , dtype = 'float64')
    lgf_eff_p           = np.zeros(n_sim_periods , dtype = 'float64')
    dlf_eff_p           = np.zeros(n_sim_periods , dtype = 'float64')
    dlf_wool_p          = np.zeros(n_sim_periods , dtype = 'float64')
    chill_p             = np.zeros(n_sim_periods , dtype = 'float64')


    c_srw_gw            = np.zeros(gw , dtype = 'float64')
    c_cp_gx             = np.zeros(gx , dtype = 'float64')
    c_cf_gx             = np.zeros(gx , dtype = 'float64')
    c_ci_gy             = np.zeros(gy , dtype = 'float64')
    c_cl_gy             = np.zeros(gy , dtype = 'float64')
    i_annual_cull_is    = np.zeros(d_is , dtype = 'float64')
    mrl_cs_jewbl        = np.zeros(jewbl , dtype = 'float64')
    mrl_mu_jewbl        = np.zeros(jewbl , dtype = 'float64')
    lw6_jexl            = np.zeros(jexl , dtype = 'float64')
    cr_jexyl            = np.zeros(jexyl , dtype = 'float64')
    mr_cs_jexyl         = np.zeros(jexyl , dtype = 'float64')
    mrt_cs_jexyl        = np.zeros(jexyl , dtype = 'float64')
    mrd_cs_jexyl        = np.zeros(jexyl , dtype = 'float64')
    mr_mu_jexyl         = np.zeros(jexyl , dtype = 'float64')
    mrt_mu_jexyl        = np.zeros(jexyl , dtype = 'float64')
    mrd_mu_jexyl        = np.zeros(jexyl , dtype = 'float64')
    nw_jexyl            = np.zeros(jexyl , dtype = 'float64')
    lw_ffcf_start_jexyl = np.zeros(jexyl , dtype = 'float64')
    aw_start_jexyl      = np.zeros(jexyl , dtype = 'float64')
    mw_start_jexyl      = np.zeros(jexyl , dtype = 'float64')
    bw_start_jexyl      = np.zeros(jexyl , dtype = 'float64')
    relsize_jexyl       = np.zeros(jexyl , dtype = 'float64')
    relsize1_jexyl      = np.zeros(jexyl , dtype = 'float64')
    rc_jexyl            = np.zeros(jexyl , dtype = 'float64')
    cfw_start_jexyl     = np.zeros(jexyl , dtype = 'float64')
    fl_start_jexyl      = np.zeros(jexyl , dtype = 'float64')
    fd_start_jexyl      = np.zeros(jexyl , dtype = 'float64')
    fd_min_start_jexyl  = np.zeros(jexyl , dtype = 'float64')
    ldr_jexyl           = np.zeros(jexyl , dtype = 'float64')
    lb_jexyl            = np.zeros(jexyl , dtype = 'float64')
    meme_jexyl          = np.zeros(jexyl , dtype = 'float64')
    level_jexyl         = np.zeros(jexyl , dtype = 'float64')
    d_lw_f_jexyl        = np.zeros(jexyl , dtype = 'float64')
    cw_jexyl            = np.zeros(jexyl , dtype = 'float64')
    mec_jexyl           = np.zeros(jexyl , dtype = 'float64')
    ldr_jexyl           = np.zeros(jexyl , dtype = 'float64')
    lb_jexyl            = np.zeros(jexyl , dtype = 'float64')
    mel_jexyl           = np.zeros(jexyl , dtype = 'float64')
    d_cfw_wolag_jexyl   = np.zeros(jexyl , dtype = 'float64')
    d_cfw_jexyl         = np.zeros(jexyl , dtype = 'float64')
    mew_jexyl           = np.zeros(jexyl , dtype = 'float64')
    d_fd_jexyl          = np.zeros(jexyl , dtype = 'float64')
    d_fl_jexyl          = np.zeros(jexyl , dtype = 'float64')
    mecold_jexyl        = np.zeros(jexyl , dtype = 'float64')
    kg_jexyl            = np.zeros(jexyl , dtype = 'float64')
    ebg_jexyl           = np.zeros(jexyl , dtype = 'float64')
    pg_jexyl            = np.zeros(jexyl , dtype = 'float64')
    lw_ffcf_max_jexyl   = np.zeros(jexyl , dtype = 'float64')
    fw_end_jexyl        = np.zeros(jexyl , dtype = 'float64')
    cfw_jexyl           = np.zeros(jexyl , dtype = 'float64')
    fl_jexyl            = np.zeros(jexyl , dtype = 'float64')
    fd_min_jexyl        = np.zeros(jexyl , dtype = 'float64')
    fd_jexyl            = np.zeros(jexyl , dtype = 'float64')
    ldr_end_jexyl       = np.zeros(jexyl , dtype = 'float64')
    lb_end_jexyl        = np.zeros(jexyl , dtype = 'float64')
    i_cull_drys_jo      = np.zeros(jo , dtype = 'float64')
    a_g0_jo             = np.zeros(jo , dtype = 'float64')
    a_g2_jo             = np.zeros(jo , dtype = 'float64')
    i_annual_cull_js    = np.zeros(js , dtype = 'float64')
    mr_cs_jwexyl        = np.zeros(jwexyl , dtype = 'float64')
    mr_mu_jwexyl        = np.zeros(jwexyl , dtype = 'float64')
    nw_jwexyl           = np.zeros(jwexyl , dtype = 'float64')
    lw_ffcf_start_jwexyl = np.zeros(jwexyl , dtype = 'float64')
    aw_start_jwexyl     = np.zeros(jwexyl , dtype = 'float64')
    mw_start_jwexyl     = np.zeros(jwexyl , dtype = 'float64')
    bw_start_jwexyl     = np.zeros(jwexyl , dtype = 'float64')
    relsize_jwexyl      = np.zeros(jwexyl , dtype = 'float64')
    relsize1_jwexyl     = np.zeros(jwexyl , dtype = 'float64')
    rc_jwexyl           = np.zeros(jwexyl , dtype = 'float64')
    cfw_start_jwexyl    = np.zeros(jwexyl , dtype = 'float64')
    fl_start_jwexyl     = np.zeros(jwexyl , dtype = 'float64')
    fd_start_jwexyl     = np.zeros(jwexyl , dtype = 'float64')
    fd_min_start_jwexyl = np.zeros(jwexyl , dtype = 'float64')
    meme_jwexyl         = np.zeros(jwexyl , dtype = 'float64')
    level_jwexyl        = np.zeros(jwexyl , dtype = 'float64')
    d_lw_f_jwexyl       = np.zeros(jwexyl , dtype = 'float64')
    cw_jwexyl           = np.zeros(jwexyl , dtype = 'float64')
    mec_jwexyl          = np.zeros(jwexyl , dtype = 'float64')
    ldr_jwexyl          = np.zeros(jwexyl , dtype = 'float64')
    lb_jwexyl           = np.zeros(jwexyl , dtype = 'float64')
    mel_jwexyl          = np.zeros(jwexyl , dtype = 'float64')
    d_cfw_wolag_jwexyl  = np.zeros(jwexyl , dtype = 'float64')
    d_cfw_jwexyl        = np.zeros(jwexyl , dtype = 'float64')
    mew_jwexyl          = np.zeros(jwexyl , dtype = 'float64')
    d_fd_jwexyl         = np.zeros(jwexyl , dtype = 'float64')
    d_fl_jwexyl         = np.zeros(jwexyl , dtype = 'float64')
    mecold_jwexyl       = np.zeros(jwexyl , dtype = 'float64')
    kg_jwexyl           = np.zeros(jwexyl , dtype = 'float64')
    ebg_jwexyl          = np.zeros(jwexyl , dtype = 'float64')
    pg_jwexyl           = np.zeros(jwexyl , dtype = 'float64')
    lw_ffcf_max_jwexyl  = np.zeros(jwexyl , dtype = 'float64')
    fw_end_jwexyl       = np.zeros(jwexyl , dtype = 'float64')
    cfw_jwexyl          = np.zeros(jwexyl , dtype = 'float64')
    fl_jwexyl           = np.zeros(jwexyl , dtype = 'float64')
    fd_min_jwexyl       = np.zeros(jwexyl , dtype = 'float64')
    fd_jwexyl           = np.zeros(jwexyl , dtype = 'float64')
    ldr_end_jwexyl      = np.zeros(jwexyl , dtype = 'float64')
    lb_end_jwexyl       = np.zeros(jwexyl , dtype = 'float64')
    foo_jxyl            = np.zeros(jxyl , dtype = 'float64')
    dmd_jxyl            = np.zeros(jxyl , dtype = 'float64')
    md_jxyl             = np.zeros(jxyl , dtype = 'float64')
    hf_jxyl             = np.zeros(jxyl , dtype = 'float64')
    a_j_kd              = np.zeros(kd , dtype = 'float64')
    a_g_kd              = np.zeros(kd , dtype = 'float64')
    a_g1_kd             = np.zeros(kd , dtype = 'float64')
    a_g0_kd             = np.zeros(kd , dtype = 'float64')
    a_w_kd              = np.zeros(kd , dtype = 'float64')
    foo_kdwbl           = np.zeros(kdwbl , dtype = 'float64')
    dmd_kdwbl           = np.zeros(kdwbl , dtype = 'float64')
    md_kdwbl            = np.zeros(kdwbl , dtype = 'float64')
    hf_kdwbl            = np.zeros(kdwbl , dtype = 'float64')
    mr_cs_kdwebl        = np.zeros(kdwebl , dtype = 'float64')
    mr_mu_kdwebl        = np.zeros(kdwebl , dtype = 'float64')
    nw_kdwebl           = np.zeros(kdwebl , dtype = 'float64')
    lw_ffcf_start_kdwebl = np.zeros(kdwebl , dtype = 'float64')
    aw_start_kdwebl     = np.zeros(kdwebl , dtype = 'float64')
    mw_start_kdwebl     = np.zeros(kdwebl , dtype = 'float64')
    bw_start_kdwebl     = np.zeros(kdwebl , dtype = 'float64')
    relsize_kdwebl      = np.zeros(kdwebl , dtype = 'float64')
    relsize1_kdwebl     = np.zeros(kdwebl , dtype = 'float64')
    rc_kdwebl           = np.zeros(kdwebl , dtype = 'float64')
    cfw_start_kdwebl    = np.zeros(kdwebl , dtype = 'float64')
    fl_start_kdwebl     = np.zeros(kdwebl , dtype = 'float64')
    fd_start_kdwebl     = np.zeros(kdwebl , dtype = 'float64')
    fd_min_start_kdwebl = np.zeros(kdwebl , dtype = 'float64')
    meme_kdwebl         = np.zeros(kdwebl , dtype = 'float64')
    level_kdwebl        = np.zeros(kdwebl , dtype = 'float64')
    d_lw_f_kdwebl       = np.zeros(kdwebl , dtype = 'float64')
    cw_kdwebl           = np.zeros(kdwebl , dtype = 'float64')
    mec_kdwebl          = np.zeros(kdwebl , dtype = 'float64')
    ldr_kdwebl          = np.zeros(kdwebl , dtype = 'float64')
    lb_kdwebl           = np.zeros(kdwebl , dtype = 'float64')
    mel_kdwebl          = np.zeros(kdwebl , dtype = 'float64')
    d_cfw_wolag_kdwebl  = np.zeros(kdwebl , dtype = 'float64')
    d_cfw_kdwebl        = np.zeros(kdwebl , dtype = 'float64')
    mew_kdwebl          = np.zeros(kdwebl , dtype = 'float64')
    d_fd_kdwebl         = np.zeros(kdwebl , dtype = 'float64')
    d_fl_kdwebl         = np.zeros(kdwebl , dtype = 'float64')
    mecold_kdwebl       = np.zeros(kdwebl , dtype = 'float64')
    kg_kdwebl           = np.zeros(kdwebl , dtype = 'float64')
    ebg_kdwebl          = np.zeros(kdwebl , dtype = 'float64')
    pg_kdwebl           = np.zeros(kdwebl , dtype = 'float64')
    lw_ffcf_max_kdwebl  = np.zeros(kdwebl , dtype = 'float64')
    fw_end_kdwebl       = np.zeros(kdwebl , dtype = 'float64')
    cfw_kdwebl          = np.zeros(kdwebl , dtype = 'float64')
    fl_kdwebl           = np.zeros(kdwebl , dtype = 'float64')
    fd_min_kdwebl       = np.zeros(kdwebl , dtype = 'float64')
    fd_kdwebl           = np.zeros(kdwebl , dtype = 'float64')
    ldr_end_kdwebl      = np.zeros(kdwebl , dtype = 'float64')
    lb_end_kdwebl       = np.zeros(kdwebl , dtype = 'float64')
    i_annual_cull_ks    = np.zeros(ks , dtype = 'float64')
    a_join_p_ojel       = np.zeros(ojel , dtype = 'float64')
    a_day90_p_ojel      = np.zeros(ojel , dtype = 'float64')
    a_6weeks_p_ojel     = np.zeros(ojel , dtype = 'float64')
    a_lamb_p_ojel       = np.zeros(ojel , dtype = 'float64')
    a_wean_p_ojel       = np.zeros(ojel , dtype = 'float64')
    age_pi              = np.zeros(pi , dtype = 'float64')
    af_wool_pi          = np.zeros(pi , dtype = 'float64')
    d_cfw_ave_pi        = np.zeros(pi , dtype = 'float64')
    nw_max_pi           = np.zeros(pi , dtype = 'float64')
    feedsupply_pi       = np.zeros(pi , dtype = 'float64')
    a_s_pil             = np.zeros(pil , dtype = 'float64')
    age_pjel            = np.zeros(pjel , dtype = 'float64')
    age_f_pjel          = np.zeros(pjel , dtype = 'float64')
    pimi_pjel           = np.zeros(pjel , dtype = 'float64')
    af_wool_pjel        = np.zeros(pjel , dtype = 'float64')
    ra_pjel             = np.zeros(pjel , dtype = 'float64')
    age_y_adj_pjel      = np.zeros(pjel , dtype = 'float64')
    mm_pjel             = np.zeros(pjel , dtype = 'float64')
    d_cfw_ave_pjel      = np.zeros(pjel , dtype = 'float64')
    a_join_o_pjel       = np.zeros(pjel , dtype = 'float64')
    a_join_o_pjel       = np.zeros(pjel , dtype = 'float64')
    a_join_o_pjel       = np.zeros(pjel , dtype = 'float64')
    a_join_o_pjel       = np.zeros(pjel , dtype = 'float64')
    a_join_o_pjel       = np.zeros(pjel , dtype = 'float64')
    a_join_o_pjel       = np.zeros(pjel , dtype = 'float64')
    age_pjl             = np.zeros(pjl , dtype = 'float64')
    af_wool_pjl         = np.zeros(pjl , dtype = 'float64')
    d_cfw_ave_pjl       = np.zeros(pjl , dtype = 'float64')
    a_s_pjl             = np.zeros(pjl , dtype = 'float64')
    nw_max_pjxyl        = np.zeros(pjxyl , dtype = 'float64')
    feedsupply_pjxyl    = np.zeros(pjxyl , dtype = 'float64')
    feedsupply_pkdwbl   = np.zeros(pkdwbl , dtype = 'float64')
    nw_max_pkdwebl      = np.zeros(pkdwebl , dtype = 'float64')
    a_s_pkdwl           = np.zeros(pkdwl , dtype = 'float64')
    age_pkel            = np.zeros(pkel , dtype = 'float64')
    af_wool_pkel        = np.zeros(pkel , dtype = 'float64')
    d_cfw_ave_pkel      = np.zeros(pkel , dtype = 'float64')
    foo_std_pr          = np.zeros(pr , dtype = 'float64')
    dmd_std_pr          = np.zeros(pr , dtype = 'float64')


    ################
    ### map inputs #
    ################
    ## map the sensitivity adjusted Excel data into the numpy arrays
    i_sf = uinp.propertydata['ExcelName']
    i_cull_drys_jo = pinp.propertydata['ExcelName']
    i_annual_cull_is = pinp.propertydata['ExcelName']
    i_annual_cull_js = pinp.propertydata['ExcelName']
    i_annual_cull_ks = pinp.propertydata['ExcelName']

    ### _map the genotype information to the _g arrays
    ... = sfun.genotype(uinp. sheep_parameters, a_c_g0, a_maternal_g0_g1
                        , a_paternal_g0_g1, a_maternal_g1_g2, a_paternal_g0_g2)

    ###########################
    ### non-loop calculations #
    ###########################
    '''Calculations for which the inputs do not depend on previous periods
    See spreadsheet: Group independent and Age,Date,Timing'''

    doy_p =
    lgf_eff_p =
    dlf_eff_p =
    dlf_wool_p =
    chill_p =
    kw =
    kc =
    birth_date_i =
    birth_date_jl =
    birth_date_jel =
    birth_date_kel =
    age_pi =
    age_pjl =
    age_pjel =
    age_pkel =
    age_f_pjel =
    age_f_pjel =
    pimi_pjel =
    ra_pjel =
    age_y_adj_pjel =
    af_wool_pi =
    af_wool_pjl =
    af_wool_pjel =
    af_wool_pkel =
    mm_pjel =
    d_cfw_ave_pi =
    d_cfw_ave_pjl =
    d_cfw_ave_pjel =
    d_cfw_ave_pkel =
    nw_max_pi =
    nw_max_pjxyl =
    nw_max_pkdwebl =

    #####################################
    ### populate the association arrays #
    #####################################
    ## the association arrays relate the slices of one array with the slices of another array
    ##needs to be within the loop because the genotype inputs can change in exp.xlsx
a_j_kd
a_c_g0
a_maternal_g0_g1
a_paternal_g0_g1
a_maternal_g1_g2
a_paternal_g0_g2
a_g_i
a_g_j
a_g0_jo
a_g2_jo
a_g_kd
a_g1_kd
a_g0_kd
a_w_i
a_w_j
a_w_kd
a_join_o_pjel
a_s_pil
a_s_pjl
a_s_pkdwl
a_join_p_ojel
a_day90_p_ojel
a_6weeks_p_ojel
a_lamb_p_ojel
a_wean_p_ojel

    ### _feed inputs
    sfun.feed_inputs function


    ##########################################
    ### Initialise then loop through periods #
    ##########################################
    ## initialise the arrays for the first period #
    lw_ffcf = i.weaning_wt
    mw = 0.7 * lw_ffcf
    aw = 0.2 * lw_ffcf
    bw = 0.1 * lw_ffcf
    cfw = 0.6 #cfw at weaning
    fd = 19 #fd at weaning
    fl = 10 #fl at weaning
    #set all arrays that are assigned using += to 0.

    ## Loop through each week of the simulation (p) for ewes
    ## # number of periods is a fixed value so I'm thinking a 'for' loop
    for p in range(n_sim_periods):
        if p != 0:  # only carry this out with p<>0
            ### _conception
            cr_ojexyl[mask] += sfun.conception(lw_ffcf[p,...], srw_j)[mask]
            # with a mask to a
            nlb_ojewbl += cr_ojexyl#convert conception in _xy format to _wb
            ### _mortality
            mr[p,...] = sfun.mortality(rc[p-1,...])
            tem[p,...], dmr[p,...], lmr[p,...] = sfun.ewe_mortality()
            nlw_ojewbl = nlb_ojewbl &
            ### _start numbers & weight
            number[p,...] = sfun.transfers(number[p-1,...], sales
                            , ewe_mortality, cr, lamb_mortality, ....)  #function call or in global
            number[p] = (number[p-1] - sales[p-1]) * (1 - mortality) ....
            lw_ffcf[p,...], mw, aw, bw, zf1, zf2 = sfun.start_weight(lw_ffcf[p-1],...)
        ### feed supply loop
        # this loop is only required if a LW target is specified for the animals
        # if there is a target then the loop needs to continue until
        # the feed supply has converged on a value that generates a liveweight
        # change close to the target
        # The loop needs to execute at least once, then repeat if there
        # is a target and the result is not close enough to the target
        if this period (p) is a new feed variation period (f) or a new MIDAS feed period (n):
            then feed_supply_jxyl = feed_supply_pjxyl[p,...]
            otherwise use feedsupply from last period (which was optimised for the target)
        Feed supply loop start
            # the loop will be a bit tricky because the target is for an array of values
            # and some parts of the array may be within the tolerance but other parts are not.
            # To further complicate it the target will often be associated with
            # the weighted average of a slice of the array rather than an individual
            # element.
            foo, dmd, supp = sfun.feed_supply(feed_supply_jxyl, foo_std, dmd_std)
            #'
            pi_jexyl = sfun.p_intake(rc, srw, rel_size)
            ri_jexyl = sfun.r_intake(foo, dmd, supp)
            mei_jexyl = pi_jexyl - np.newaxis(e, supp_jxyl) * ri_jexyl * nv_jexyl + newaxis(supp_jxyl) * supp_md
            p_mei_pjexyl[p,...] = mei_jexyl
            mem = sfun.energy(....)
            mep, cw = sfun.pregnancy(....)
            mel = sfun.lactation(....)
            dcfw, new = sfun.wool_growth(....)
            ebg, pg = sfun.lw_change(mei, mem, mep, mel, mew, mecold, wmax, zf1, zf2)
            lwc = ebg * (1)
            if there is a target and abs(lwc-target) > eps:
                update feed_supply
                #      feed supply is a number between 0 and 3. We could use a binary
                #      type process to converge on the feed supply. But given that
                #      the feed supply was calculated in the previous period and
                #      it should be close then maybe a step process might be quicker.
                #      The main advantage of the binary approach is that each element
                #      of the array should converge at a similar rate, whereas maybe
                #      not with the step approach
                #      Open to ideas here.
            loop if feed_supply was changed
        lw_ffcf_jexyl = lw_ffcf_start_jexyl + lwc_jexyl * step
        lw_ffcf_max_jexyl = np.maximum(lw_ffcf_jexyl, lw_ffcf_max_jexyl)
        aw_jexyl
        mw_jexyl
        bw_jexyl
        ww_jexyl
        gw_jexyl
        fw_end_jexyl
        cfw_jexyl = cfw_start_jexyl + dcfw * step
        fl_jexyl
        fd_min_jexyl
        fd_jexyl
        ldr_end_jexyl
        lb_end_jexyl
        lw_jexyl = lw_ffcf_jexyl + cw_jexyl + cfw_Jexyl
        r_lw_jexyl[p,...] = lw_jexyl


    # repeat loop for rams & then for offspring
    # these don't require conception, pregnancy, lactation and ewe mortality
    for p in range(n_sim_periods):
        if p <>0:  # only carry this out with p<>0
            ## or pass lw_cfff_end and nw_end & srw and calculate z and rc
            mr[p,...] = sfun.mortality(rc[p-1,...])   # offspring
            mr[p,...] = sfun.mortality(rc[p-1,...])   # rams
            .... = sfun.numbers(....)                 #offspring
            .... = sfun.numbers(....)                 #rams
            lw_ffcf[p,...], mw, aw, bw, zf1, zf2 = sfun.start_weight(lw_ffcf[p-1],...)
            lw_ffcf[p,...], mw, aw, bw, zf1, zf2 = sfun.start_weight(lw_ffcf[p-1],...)
        Feed supply Loop for offspring
            #` mei and rc are not defined
            mei[p,...] = sfun.intake(rc, c_ci_gy, )
            mem = sfun.energy(....)
            dcfw, new = sfun.wool_growth(....)
            cfw = cfw_start + dcfw
            wmax = np.maximum(lw_ffcf,axis=0)
            lwc = sfun.lw_change(mei[p,...], mem, new, wmax, zf1, zf2)
            .... = sfun.end_values
        Feed supply Loop for rams #Probably will never need to loop this
            #because not specifying a target for the rams
            mei[p,...] = sfun.intake(....)
            mem = sfun.energy(....)
            dcfw, new = sfun.wool_growth(....)
            cfw = cfw_start + dcfw
            wmax = np.maximum(lw_ffcf,axis=0)
            lwc = sfun.lw_change(mei[p,...], mem, new, wmax, zf1, zf2)
            .... = sfun.end_values

def parameters():
    """Parameter generation for the pyomo variables


    Returns
    -------
    dictionaries for pyomo
    """
parameters = np.zeros((len(output_required),len(activities0)), dtype = 'float64')
    # Loop through the number of variables
    for a in activites:
        ### create array masks  for the pyomo variable
        ''' For each pyomo variable create a mask that represents the animals
        The arrays can then be summed across the axes for that mask '''
        mask = sfun.create_mask(i_activity_definition)

        ### apply each mask to each simulation output
        #output_required is a list of the arrays that are required as parameters
        for n, o in enumerate(output_required):
            parameters[n,a] = np.sum(o[mask])

return parameters

''' Or to allow one function call per constraint this function could
generate the array and then multiple functions that just return the
required row of the array.'''