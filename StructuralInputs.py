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

import Functions as fun

#########################################################################################################################################################################################################
#########################################################################################################################################################################################################
# read in excel
#########################################################################################################################################################################################################
#########################################################################################################################################################################################################

##build path this way so that readthedocs can read correctly.
directory_path = os.path.dirname(os.path.abspath(__file__))
structural_xl_path = os.path.join(directory_path, "Structural.xlsx")
structural_pkl_path = os.path.join(directory_path, "pkl_structural.pkl")


try:
    if os.path.getmtime(structural_xl_path) > os.path.getmtime(structural_pkl_path):
        inputs_from_pickle = False
    else:
        inputs_from_pickle = True
        print('Reading structural inputs from pickle',end=' ',flush=True)
except FileNotFoundError:
    inputs_from_pickle = False

##if inputs are not read from pickle then they are read from excel and written to pickle
if inputs_from_pickle == False:
    print('Reading structural inputs from Excel',end=' ',flush=True)
    with open(structural_pkl_path,"wb") as f:
        ##general
        general_inp = fun.xl_all_named_ranges(structural_xl_path,"General")
        pkl.dump(general_inp,f,protocol=pkl.HIGHEST_PROTOCOL)

        ##sheep inputs
        stock_inp = fun.xl_all_named_ranges(structural_xl_path,'Stock',numpy=True)
        pkl.dump(stock_inp,f,protocol=pkl.HIGHEST_PROTOCOL)

        ##sa inputs (these variables can have sensitivity applied from exp.xl
        structuralsa_inp = fun.xl_all_named_ranges(structural_xl_path,'StructuralSA',numpy=True)
        pkl.dump(structuralsa_inp,f,protocol=pkl.HIGHEST_PROTOCOL)

##else the inputs are read in from the pickle file
##note this must be in the same order as above
else:
    with open(structural_pkl_path,"rb") as f:
        general_inp = pkl.load(f)
        stock_inp = pkl.load(f)
        structuralsa_inp = pkl.load(f)

print('- finished')

##reshape require inputs
###lengths
len_b0 = stock_inp['ia_ppk5_lsb0'].shape[-1]
len_b1 = stock_inp['ia_ppk2g1_rlsb1'].shape[-1]
len_l = stock_inp['i_len_l']
len_m = stock_inp['i_len_m']
len_s = stock_inp['i_len_s']
len_r = stock_inp['i_n_r1type']

###shapes
rlsb1 = (len_r,len_l,len_s,len_b1)
mlsb1 = (len_m,len_l,len_s,len_b1)
lsb0 = (len_l,len_s,len_b0)

###stock
stock_inp['ia_ppk2g1_rlsb1'] = np.reshape(stock_inp['ia_ppk2g1_rlsb1'],rlsb1)
stock_inp['ia_ppk5_lsb0'] = np.reshape(stock_inp['ia_ppk5_lsb0'],lsb0)
stock_inp['ia_k2_mlsb1'] = np.reshape(stock_inp['ia_k2_mlsb1'],mlsb1)



##create a copy of each input dict - so that the base inputs remain unchanged
##the copy created is the one used in the actual modules
###NOTE: if an input sheet is added remember to add it to the dict reset in f_sa() below.
general = copy.deepcopy(general_inp)
stock = copy.deepcopy(stock_inp)
structuralsa = copy.deepcopy(structuralsa_inp)


#######################
#apply SA             #
#######################
def f_structural_inp_sa():
    '''
    Applies sensitivity adjustment to relevant inputs. Note only inputs in StructuralSA sheet can have sensitivities applied.
    After the sensitivities are applied, when using the DSP model, inputs with a feed period index are expanded to
    account for additional feed periods that are added due to season nodes.
    This function gets called at the beginning of each loop in the exp.py module

    :return: None.

    '''
    ##have to import it here since sen.py imports this module
    import Sensitivity as sen

    ##reset inputs to base at the start of each trial before applying SA  - old method was to update the SA based on the _inp dict but that doesn't work well when multiple SA on the same variale.
    fun.f_dict_reset(structuralsa, structuralsa_inp)


    ##SAV
    structuralsa['i_nut_spread_n1'] = fun.f_sa(structuralsa['i_nut_spread_n1'], sen.sav['nut_spread_n1'],5)
    structuralsa['i_nut_spread_n3'] = fun.f_sa(structuralsa['i_nut_spread_n3'], sen.sav['nut_spread_n3'],5)
    structuralsa['i_n1_len'] = fun.f_sa(structuralsa['i_n1_len'], sen.sav['n_fs_dams'],5)
    structuralsa['i_n3_len'] = fun.f_sa(structuralsa['i_n3_len'], sen.sav['n_fs_offs'],5)
    structuralsa['i_w_start_len1'] = fun.f_sa(structuralsa['i_w_start_len1'], sen.sav['n_initial_lw_dams'],5)
    structuralsa['i_adjp_lw_initial_w1'] = fun.f_sa(structuralsa['i_adjp_lw_initial_w1'], sen.sav['adjp_lw_initial_w1'],5)
    structuralsa['i_fvp_mask_dams'] = fun.f_sa(structuralsa['i_fvp_mask_dams'], sen.sav['mask_fvp_dams'],5)
    structuralsa['i_dvp_mask_f1'] = fun.f_sa(structuralsa['i_dvp_mask_f1'], sen.sav['fvp_is_dvp_dams'],5)
    structuralsa['i_fvp_mask_offs'] = fun.f_sa(structuralsa['i_fvp_mask_offs'], sen.sav['mask_fvp_offs'],5)
    structuralsa['i_dvp_mask_f3'] = fun.f_sa(structuralsa['i_dvp_mask_f3'], sen.sav['fvp_is_dvp_offs'],5)
    structuralsa['i_rev_create'] = fun.f_sa(structuralsa['i_rev_create'], sen.sav['rev_create'],5)
    structuralsa['i_rev_number'] = fun.f_sa(structuralsa['i_rev_number'], sen.sav['rev_number'],5)
    structuralsa['i_rev_trait_inc'] = fun.f_sa(structuralsa['i_rev_trait_inc'], sen.sav['rev_trait_inc'],5)
    structuralsa['i_generate_with_t'] = fun.f_sa(structuralsa['i_generate_with_t'], sen.sav['gen_with_t'],5)
    structuralsa['i_fs_create'] = fun.f_sa(structuralsa['i_fs_create'], sen.sav['fs_create'],5)
    structuralsa['i_fs_use_pkl'] = fun.f_sa(structuralsa['i_fs_use_pkl'], sen.sav['fs_use_pkl'],5)
    structuralsa['i_fs_number'] = fun.f_sa(structuralsa['i_fs_number'], sen.sav['fs_number'],5)
    structuralsa['i_r2adjust_inc'] = fun.f_sa(structuralsa['i_r2adjust_inc'], sen.sav['r2adjust_inc'],5)

##############################
# handle inputs with p6 axis #
##############################
def f1_expand_p6():
    ##When using DSP, expand inputs with a p6 axis for each season node.
    ##has to be a separate function to the sa because values altered in SA impact a_p6std_p6z
    ##have to import it here since sen.py imports this module
    import Periods as per

    ###get association between the input fp and the node adjusted fp
    a_p6std_p6z = per.f_feed_periods(option=2)
    ###apply association
    ####stock
    structuralsa['i_nv_upper_p6z'] = np.take_along_axis(structuralsa['i_nv_upper_p6'][:,None], a_p6std_p6z, axis=0)
    structuralsa['i_nv_lower_p6z'] = np.take_along_axis(structuralsa['i_nv_lower_p6'][:,None], a_p6std_p6z, axis=0)



##############
#phases      #
##############
def f_phases():
    ##rotation phases and constraints read in from excel
    return pd.read_excel('Rotation.xlsx', sheet_name='rotation list', header= None, index_col = 0, engine='openpyxl').T.reset_index(drop=True).T  #reset the col headers to std ie 0,1,2 etc



###############
#landuse sets #
###############
landuse = {}

##landuse indexes
landuse['All']=general['i_idx_k'] #used in reporting and bounds and as index in precalc modules
landuse['C']=general['i_idx_k1'] #all crops, used in stubble and mach (not used for rotations)
landuse['All_pas']=general['i_idx_k2'] #used in reporting

##next set is used in pasture.py for mobilisation of below ground reserves and phase area
landuse['pasture_sets']={'annual': {'a', 'ar'
                                , 's', 'sr'
                                , 'm'}
                        ,'lucerne':{'u', 'uc', 'ur'
                                   , 'x', 'xc', 'xr'}
                        ,'tedera':{'j','jc', 't','tc', 'jr', 'tr'}
                       }

##A1, E, P, G and C1 are just used in pas.py for germination ^can be removed when/if germination is calculated from sim
landuse['G']={'b', 'bd', 'h', 'o', 'od', 'of', 'w', 'wd', 'f','i', 'k', 'l', 'v', 'z', 'zd', 'r', 'rd'
                , 'a', 'ar'
                , 's', 'sr'
                , 'm'
                , 'u', 'ur'
                , 'x', 'xr'
                , 'j', 't', 'jr', 'tr'
                , 'G', 'Y', 'B','O','W', 'N', 'L', 'F', 'OF'
                , 'A', 'AR'
                , 'S', 'SR'
                , 'M'
                , 'U'
                , 'X'
                , 'T', 'J'} #all landuses
landuse['C1']={'B','O','W', 'N', 'L', 'F', 'OF', 'b', 'bd', 'h', 'o', 'od', 'of', 'w', 'wd', 'f','i', 'k', 'l', 'v', 'z', 'zd', 'r', 'rd'} #all crops - had to create a separate set because don't want the capital in the crop set above as it is used to create pyomo set
landuse['P']={'L', 'F', 'f','i', 'k', 'l', 'v'} #pulses
landuse['E']={'B','O','W', 'OF', 'b', 'bd', 'h', 'o', 'od', 'of', 'w', 'wd'} #cereals
landuse['A1']={'a', 's', 'm'} #annual not resown - special set used in pasture germ and con2 when determining if a rotation provides a rotation because in yr1 we don't want ar to provide an A because we need to distinguish between them

##dry sown crops, used in phase.py for seeding param (not used for building rotations)
landuse['dry_sown'] = {'bd', 'od', 'wd', 'zd','rd'}

##all crops that produce hay - used in machpyomo/coremodel for hay con
landuse['Hay']={'h'}


##########################################
#Landuse sets used in to build rotations #
##########################################
landuse['A']={'a', 'A', 'AR'
                , 'S'
                , 'M'} #annual
landuse['AR']={'ar', 'AR'} #resown annual
landuse['B']={'B', 'b', 'bd'} #barleys
landuse['J']={'J', 'j', 'jr'} #tedera
landuse['M']={'m', 'M'} #manipulated pasture
landuse['N']={'N', 'z', 'zd', 'r', 'rd'} #canolas
landuse['O']={'O', 'OF', 'h', 'o', 'od'} #oats
landuse['OF']={'OF', 'of'} #oats fodder
landuse['F']={'F', 'f'} #faba
landuse['L']={'L', 'l'} #lupin
landuse['S1']={'S1','s'} #spray topped pasture yr1 - needs to be numbered so that s cannot provide A in yr1
landuse['SR1']={'SR1','sr'} #spray topped pasture yr1
landuse['S']={'S','S1','SR1'} #spray topped pasture yr1
landuse['T']={'T', 't', 'tr','J', 'j', 'jr'} #tedera - also includes manipulated tedera because it is combined in yrs 3,4,5
landuse['W']={'W', 'w', 'wd'} #wheats
landuse['U']={'u', 'ur', 'U','x', 'xr', 'X'} #lucerne
landuse['X']={'x', 'xr', 'X'} #lucerne
landuse['Y']={'b', 'bd', 'h', 'o', 'od', 'of', 'w', 'wd', 'f','i', 'k', 'l', 'v', 'z', 'zd', 'r', 'rd'
                , 'Y', 'B','O','W', 'N', 'L', 'F', 'OF'} #anything not pasture

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
