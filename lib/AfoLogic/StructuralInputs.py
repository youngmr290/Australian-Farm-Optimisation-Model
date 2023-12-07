"""

Contains all the structural inputs relating to the operation of the model.

The inputs are read in from Structural.xlsx.

.. note::
    Only to be changed by experienced AFO developers.

author: young
"""

##python modules (CAN'T import PropertyInputs)
import pickle as pkl
import pandas as pd
import numpy as np
import copy
import os.path

from . import Functions as fun

def f_reshape_sinp_defaults(sinp_defaults):
    ##reshape require inputs
    ###lengths
    len_b0 = sinp_defaults["stock_inp"]['ia_ppk5_lsb0'].shape[-1]
    len_b1 = sinp_defaults["stock_inp"]['ia_ppk2g1_rlsb1'].shape[-1]
    len_l = sinp_defaults["stock_inp"]['i_len_l']
    len_m = sinp_defaults["stock_inp"]['i_len_m']
    len_s = sinp_defaults["stock_inp"]['i_len_s']
    len_r = sinp_defaults["stock_inp"]['i_n_r1type']

    ###shapes
    rlsb1 = (len_r,len_l,len_s,len_b1)
    mlsb1 = (len_m,len_l,len_s,len_b1)
    lsb0 = (len_l,len_s,len_b0)

    ###stock
    sinp_defaults["stock_inp"]['ia_ppk2g1_rlsb1'] = np.reshape(sinp_defaults["stock_inp"]['ia_ppk2g1_rlsb1'],rlsb1)
    sinp_defaults["stock_inp"]['ia_ppk5_lsb0'] = np.reshape(sinp_defaults["stock_inp"]['ia_ppk5_lsb0'],lsb0)
    sinp_defaults["stock_inp"]['ia_k2_mlsb1'] = np.reshape(sinp_defaults["stock_inp"]['ia_k2_mlsb1'],mlsb1)


#######################
#reset defaults       #
#######################
def f_select_n_reset_sinp(sinp_defaults):
    ##occurs for each trial
    ##create a copy of each input dict - so that the base inputs remain unchanged
    ##the copy created is the one used in the actual modules
    ###NOTE: if an input sheet is added remember to add it to the dict reset in f_sa() below.
    global general
    global stock
    global structuralsa
    global rep
    general = copy.deepcopy(sinp_defaults["general_inp"])
    stock = copy.deepcopy(sinp_defaults["stock_inp"])
    structuralsa = copy.deepcopy(sinp_defaults["structuralsa_inp"])
    rep = copy.deepcopy(sinp_defaults["rep_inp"])


#######################
#apply SA             #
#######################
def f_structural_inp_sa(sinp_defaults):
    '''
    Applies sensitivity adjustment to relevant inputs. Note only inputs in StructuralSA sheet can have sensitivities applied.
    After the sensitivities are applied, when using the DSP model, inputs with a feed period index are expanded to
    account for additional feed periods that are added due to season nodes.
    This function gets called at the beginning of each loop in the exp.py module

    :return: None.

    '''
    ##have to import it here since sen.py imports this module
    from . import Sensitivity as sen

    ##SAV
    structuralsa['i_nut_spread_n1'] = fun.f_sa(structuralsa['i_nut_spread_n1'], sen.sav['nut_spread_n1'],5)
    structuralsa['i_confinement_n1'] = fun.f_sa(structuralsa['i_confinement_n1'], sen.sav['confinement_n1'],5)
    structuralsa['i_nut_spread_n3'] = fun.f_sa(structuralsa['i_nut_spread_n3'], sen.sav['nut_spread_n3'],5)
    structuralsa['i_confinement_n3'] = fun.f_sa(structuralsa['i_confinement_n3'], sen.sav['confinement_n3'],5)
    structuralsa['i_n1_len'] = fun.f_sa(structuralsa['i_n1_len'], sen.sav['n_fs_dams'],5)
    structuralsa['i_n3_len'] = fun.f_sa(structuralsa['i_n3_len'], sen.sav['n_fs_offs'],5)
    structuralsa['i_w_start_len1'] = fun.f_sa(structuralsa['i_w_start_len1'], sen.sav['n_initial_lw_dams'],5)
    structuralsa['i_adjp_lw_initial_w1'] = fun.f_sa(structuralsa['i_adjp_lw_initial_w1'], sen.sav['adjp_lw_initial_w1'],5)
    structuralsa['i_adjp_cfw_initial_w1'] = fun.f_sa(structuralsa['i_adjp_cfw_initial_w1'], sen.sav['adjp_cfw_initial_w1'],5)
    structuralsa['i_adjp_fd_initial_w1'] = fun.f_sa(structuralsa['i_adjp_fd_initial_w1'], sen.sav['adjp_fd_initial_w1'],5)
    structuralsa['i_adjp_fl_initial_w1'] = fun.f_sa(structuralsa['i_adjp_fl_initial_w1'], sen.sav['adjp_fl_initial_w1'],5)
    structuralsa['i_fvp_mask_dams'] = fun.f_sa(structuralsa['i_fvp_mask_dams'], sen.sav['mask_fvp_dams'],5)
    structuralsa['i_dvp_mask_f1'] = fun.f_sa(structuralsa['i_dvp_mask_f1'], sen.sav['fvp_is_dvp_dams'],5)
    structuralsa['i_fvp_mask_offs'] = fun.f_sa(structuralsa['i_fvp_mask_offs'], sen.sav['mask_fvp_offs'],5)
    structuralsa['i_dvp_mask_f3'] = fun.f_sa(structuralsa['i_dvp_mask_f3'], sen.sav['fvp_is_dvp_offs'],5)
    structuralsa['i_rev_create'] = fun.f_sa(structuralsa['i_rev_create'], sen.sav['rev_create'],5)
    structuralsa['i_rev_number'] = fun.f_sa(structuralsa['i_rev_number'], sen.sav['rev_number'],5)
    structuralsa['i_rev_trait_inc'] = fun.f_sa(structuralsa['i_rev_trait_inc'], sen.sav['rev_trait_inc'],5)
    structuralsa['i_generate_with_t'] = fun.f_sa(structuralsa['i_generate_with_t'], sen.sav['gen_with_t'],5)
    structuralsa['i_fs_create_pkl'] = fun.f_sa(structuralsa['i_fs_create_pkl'], sen.sav['fs_create_pkl'],5)
    structuralsa['i_fs_create_number'] = fun.f_sa(structuralsa['i_fs_create_number'], sen.sav['fs_create_number'],5)
    structuralsa['i_fs_use_pkl'] = fun.f_sa(structuralsa['i_fs_use_pkl'], sen.sav['fs_use_pkl'],5)
    structuralsa['i_fs_use_number'] = fun.f_sa(structuralsa['i_fs_use_number'], sen.sav['fs_use_number'],5)
    structuralsa['i_use_pkl_condensed_start_condition'] = fun.f_sa(structuralsa['i_use_pkl_condensed_start_condition'], sen.sav['use_pkl_condensed_start_condition'],5)
    structuralsa['i_r2adjust_inc'] = fun.f_sa(structuralsa['i_r2adjust_inc'], sen.sav['r2adjust_inc'],5)

    ##report controls
    ###SAV
    rep['i_store_nv_rep'] = fun.f_sa(rep['i_store_nv_rep'], sen.sav['nv_inc'], 5)
    rep['i_store_cs_rep'] = fun.f_sa(rep['i_store_cs_rep'], sen.sav['cs_inc'], 5)
    rep['i_store_fs_rep'] = fun.f_sa(rep['i_store_fs_rep'], sen.sav['fs_inc'], 5)
    rep['i_store_lw_rep'] = fun.f_sa(rep['i_store_lw_rep'], sen.sav['lw_inc'], 5)
    rep['i_store_ebw_rep'] = fun.f_sa(rep['i_store_ebw_rep'], sen.sav['ebw_inc'], 5)
    rep['i_store_on_hand_mort'] = fun.f_sa(rep['i_store_on_hand_mort'], sen.sav['onhand_mort_p_inc'], 5)
    rep['i_store_mort'] = fun.f_sa(rep['i_store_mort'], sen.sav['mort_inc'], 5)
    rep['i_store_feedbud'] = fun.f_sa(rep['i_store_feedbud'], sen.sav['feedbud_inc'], 5)


##############################
# handle inputs with p6 axis #
##############################
def f1_expand_p6():
    ##When using DSP, expand inputs with a p6 axis for each season node.
    ##has to be a separate function to the sa because values altered in SA impact a_p6std_p6z
    ##have to import it here since sen.py imports this module
    from . import Periods as per

    ###get association between the input fp and the node adjusted fp
    a_p6std_p6z = per.f_feed_periods(option=2)
    ###apply association
    ####stock
    structuralsa['i_nv_upper_p6z'] = np.take_along_axis(structuralsa['i_nv_upper_p6'][:,None], a_p6std_p6z, axis=0)
    structuralsa['i_nv_lower_p6z'] = np.take_along_axis(structuralsa['i_nv_lower_p6'][:,None], a_p6std_p6z, axis=0)


###############
#landuse sets #
###############
def f_landuse_sets():
    global landuse
    landuse = {}

    ##landuse indexes
    landuse['All']=general['i_idx_k'] #used in reporting and bounds and as index in precalc modules
    landuse['C']=general['i_idx_k1'] #all crops, used in stubble and mach (not used for rotations)
    landuse['All_pas']=general['i_idx_k2'] #used in reporting

    ##next set is used in pasture.py for mobilisation of below ground reserves and phase area
    landuse['pasture_sets']={'annual': {'a', 'ar', 'a2'
                                    , 's', 'sr'
                                    , 'm'}
                            ,'lucerne':{'u', 'uc', 'ur'
                                       , 'x', 'xc', 'xr'}
                            ,'tedera':{'j','jc', 't','tc', 'jr', 'tr'}
                            ,'understory':{'sp'}
                           }

    ##next set is used in rotation.py to identify which sown pasture landuses can be dry sown. Used in the handling of rotations (dry sown landuses can be incremented before the brk of season).
    landuse['resown_pasture_sets']={'annual': {'ar', 'sr'}
                            ,'lucerne':{'ur', 'xr'}
                            ,'tedera':{'jr', 'tr'}
                            ,'understory':{}
                           }

    ##A1, E, P, G and C1 are just used in pas.py for germination ^can be removed when/if germination is calculated from sim
    ## these are also used for PNC landuses. & E is used in reporting
    landuse['G']={'b', 'bd', 'h', 'o', 'od', 'of', 'w', 'wd', 'f','i', 'k', 'l', 'v', 'z', 'zd', 'r', 'rd'
                    , 'a', 'ar', 'a2'
                    , 's', 'sr'
                    , 'sp'
                    , 'm'
                    , 'u', 'ur'
                    , 'x', 'xr'
                    , 'j', 't', 'jr', 'tr'
                    , 'G', 'Y', 'B','O','O1','W', 'N', 'K', 'L', 'F', 'OF'
                    , 'A', 'A1', 'A2', 'AR'
                    , 'S', 'S1', 'SR1'
                    , 'SP'
                    , 'M'
                    , 'U'
                    , 'X'
                    , 'T', 'J'} #all landuses
    landuse['C1']={'C1','B','O','O1','W', 'N', 'K', 'L', 'F', 'OF', 'b', 'bd', 'h', 'o', 'od', 'of', 'w', 'wd', 'f','i', 'k', 'l', 'v', 'z', 'zd', 'r', 'rd'} #all crops - had to create a separate set because don't want the capital in the crop set above as it is used to create pyomo set
    landuse['P']={'P','K','L', 'F', 'f','i', 'k', 'l', 'v'} #pulses
    landuse['E']={'E','B','O','O1','W', 'OF', 'b', 'bd', 'h', 'o', 'od', 'of', 'w', 'wd'} #cereals
    landuse['Ag0']={'a', 'a2', 's', 'm'} #annual not resown - special set used in pasture germ and con2 when determining if a rotation provides a rotation because in yr1 we don't want ar to provide an A because we need to distinguish between them
    landuse['Ag1']={'Ag1', 'A1', 'AR', 'a', 'ar'} #all non-spraytopped annual sets that can exist in yr1
    landuse['Ag2']={'Ag2', 'A', 'AR', 'A2', 'A1'
                    , 'S', 'S1','SR1'
                    , 'M'} #all annual sets that can exist in yr2
    landuse['Sg1']={'Sg1', 'S','S1','SR1', 's', 'sr'} #all spraytopped annual sets that can exist in yr1

    ##dry sown crops, used in phase.py for seeding param (not used for building rotations)
    landuse['dry_sown'] = {'bd', 'od', 'wd', 'zd','rd'}

    ##all crops that produce hay - used in machpyomo/coremodel for hay con
    landuse['Hay']={'h'}


    ##########################################
    #Landuse sets used in to build rotations #
    ##########################################
    landuse['A1']={'a', 'A1'} #annual yr1
    landuse['A2']={'A2', 'A1', 'AR'} #annual yr2
    landuse['A']={'A', 'A2'
                    , 'S'
                    , 'M'} #annual
    # landuse['A']={'A', 'A3'} #annual
    landuse['AR']={'ar', 'AR'} #resown annual
    landuse['B']={'B', 'b', 'bd'} #barleys
    landuse['J']={'J', 'j', 'jr'} #tedera
    landuse['K']={'K', 'k'} #chic pea
    landuse['M']={'m', 'M'} #manipulated pasture
    landuse['N']={'N', 'z', 'zd', 'r', 'rd'} #canolas
    landuse['O1']={'O1', 'h', 'o', 'od'} #oats - only in yr1 doesnt include foder
    landuse['O']={'O', 'O1', 'OF', 'h', 'o', 'od', 'of'} #oats
    landuse['OF']={'OF', 'of'} #oats fodder
    landuse['F']={'F', 'f'} #faba
    landuse['L']={'L', 'l'} #lupin
    landuse['S1']={'S1','s'} #spray topped pasture yr1 - needs to be numbered so that sr cannot provide S in yr1
    landuse['SR1']={'SR1','sr'} #spray topped pasture yr1
    landuse['S']={'S','S1','SR1'} #spray topped pasture yr1
    landuse['SP']={'SP','sp'} #salt land pasture (can only be in a cont rotation)
    landuse['T']={'T', 't', 'tr','J', 'j', 'jr'} #tedera - also includes manipulated tedera because it is combined in yrs 3,4,5
    landuse['W']={'W', 'w', 'wd'} #wheats
    landuse['U']={'u', 'ur', 'U','x', 'xr', 'X'} #lucerne
    landuse['X']={'x', 'xr', 'X'} #lucerne
    landuse['Y']={'b', 'bd', 'h', 'o', 'od', 'of', 'w', 'wd', 'f','i', 'k', 'l', 'v', 'z', 'zd', 'r', 'rd'
                    , 'Y', 'B','O','W', 'N', 'K', 'L', 'F', 'OF'} #anything not pasture

    landuse['a']={'a'}
    landuse['ar']={'ar'}
    landuse['b']={'b'}
    landuse['bd']={'bd'}
    landuse['f']={'f'}
    landuse['h']={'h'}
    landuse['i']={'i'}
    landuse['j']={'j'}
    landuse['jc']={'jc'}
    landuse['jr']={'jr'}
    landuse['k']={'k'}
    landuse['l']={'l'}
    landuse['m']={'m'}
    landuse['o']={'o'}
    landuse['od']={'od'}
    landuse['of']={'of'}
    landuse['r']={'r'}
    landuse['rd']={'rd'}
    landuse['s']={'s'}
    landuse['sp']={'sp'}
    landuse['sr']={'sr'}
    landuse['t']={'t'}
    landuse['tc']={'tc'}
    landuse['tr']={'tr'}
    landuse['u']={'u'}
    landuse['uc']={'uc'}
    landuse['ur']={'ur'}
    landuse['v']={'v'}
    landuse['w']={'w'}
    landuse['wd']={'wd'}
    landuse['x']={'x'}
    landuse['xc']={'xc'}
    landuse['xr']={'xr'}
    landuse['z']={'z'}
    landuse['zd']={'zd'}




    ##special sets used in crop sim
    # landuse['Ys'] = {'Y'}
    # landuse['As'] = {'A','a'}
    # landuse['JR'] = {'jr'}
    # landuse['TR'] = {'tr'}
    # landuse['UR'] = {'ur'}
    # landuse['XR'] = {'xr'}
    # landuse['PAS'] = {'A', 'AR', 'S', 'SR', 'M','T','J','U','X', 'tc', 'jc', 'uc', 'xc'}


#########################################################################################################################################################################################################
#########################################################################################################################################################################################################
# functions that use data from above
#########################################################################################################################################################################################################
#########################################################################################################################################################################################################


def end_col():
    '''Specifies the column number for the current landuse in the rotation dataframe.
    Used in the crop module
    '''
    end_col  = [general['phase_len']-1]
    return end_col
