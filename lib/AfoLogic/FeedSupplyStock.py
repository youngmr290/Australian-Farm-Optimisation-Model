'''

author: young

Feedsupply is represented as a nutritive value (NV). There is also an input which controls if the animal
is in confinement or not.

Animal production is generated for animals following different liveweight profiles throughout the year. There
are a number of starting liveweights and for each starting liveweight there are a number of feedsupply patterns.
The LW profiles are not continuous for the entirety of the animal's life, rather the animals are distributed to the
starting liveweights at the beginning of each year (prejoining). This is done to reduce model size.

The standard feedsupply options throughout the year are generated from an input of the expected optimum feedsupply
and a range either side based on specified nutrition levels. The feedsupply is
input for three levels; expected optimum, higher and lower and the standard feedsupply is created by scaling
the expected optimum using the range between the higher and lower levels and the input values for nutrition level.
For each feed variation period (FVP) the number of nutrition options is determined by the specified nutrition levels.
The number of nutrition options each year depends on this number and the number of FVPs. At the start of each
FVP each nutrition option up to that point can then split based on the number of specified nutrition levels.
For example, the high nutrition feedsupply in the first FVP is followed by all feedsupply levels in the
second FVP. See below for a visual description of the feedsupply patterns.

.. image:: FS_diagram.png

Figure 1: Representation of the nutrition profile options (feedsupply patterns) included in the livestock data
generator if there were 1 starting liveweight 2 nutrition levels and 3 feed variation periods.

The standard feedsupply can be generated in two ways:

    #. from inputs in Property.xl and Structural.xl
    #. from a pkl file that is generated from the optimum nutrition in a previous trial with multiple w options.

The spreadsheet inputs for Feedsupply contain a base feedsupply (i_feedoptions_r1pj0) for different animals.
The base feed supply used for each animal class in the generator is selected using ia_r1_zig?.

For dams (has not been hooked up for offs) there is possible further adjustment (i_feedsupply_adj_options_r2p)
for LSLN (litter size lactation number - b1 axis) and weaning age (a1 axis). Typically this is not used.
Instead the user runs AFO with multiple nutrition levels and allows the model to
generate the optimal feedsupply which is stored in a pkl file and used for subsequent runs.

Generating the feedsupply from the optimal nutrition pattern selected by the model means that the feedsupply
can easily be optimised for different axis that are not included in the spreadsheet inputs (e.g. t axis and z axis).
This is carried out in the feedsupply creation experiment (in exp.xl). Depending on the number of w that
have been included in the trials it may take several iterations to converge on the optimal feedsupply.
The inclusion of confinement feeding may also be optimised and stored in the pkl file. However, optimising
the duration of confinement feeding is imprecise because the optimisation occurs at the DVP level
and this time-step is likely too long. Therefore, the user needs to test including confinement in a range of feed
periods for different stock classes. This is controlled using the input i_confinement_r1p6z.

When using the pkl file as the source of the feedsupply inputs it is possible to control whether the inclusion
of confinement is controlled by the previous optimum solution (from the pkl file) or from the
input i_confinement_r1p6z. This allows the inclusion of confinement feeding in subsequent trials even if it wasn't
selected in the previous optimum solution.
This also allows confinement to be included in the N1 model without forcing it to occur all year.

Differentiating feeding supplement in confinement from supplementary feeding in the paddock by separating
both the DV's for the livestock and a separate feed pool constraint is good for a few reason:

    #. Feeding a given quantity of energy per day in confinement results in better production because less energy
       is expended grazing.
    #. Confinement feeding should incur a capital cost of the feed lot and the labour requirement should reflect
       feedlot conditions rather than paddock feeding conditions.
    #. Restricted intake of supplement in confinement that meets the energy requirements of an animal results
       in slack volume. If the supplement fed in confinement was in the same feed pool as dry residues this slack
       volume could be used to consume greater quantities of poor quality feed such as dry pasture or stubble.
    #. The confinement pool is not adjusted for effective MEI, which is to allow for the lower efficiency of feeding
       if feed quality is greater than the quality required to meet the target for the given pool.
    #. Reason 4 should be removed because no supplement is scaled by effective mei (regardless of feed pool)
       Michael, after you have read this comment you can delete these 2 points

Likely process to calibrate feedsupply: feedsupply can be calibrated prior to an analysis using multiple n
slices and once the optimum is identified the model can be set back to N1. This might take multiple iterations
using the big model. The feedsupply will need to be generated for each group of trials that have different
active sheep axis (ie if the analysis is comparing scanning options then there would need to be a feedsupply
for scan=0 and scan=1 because the clustering is different). A further challenge is to optimise when to confinement
feed. Optimising the NV in confinement is similar to optimising in the paddock and requires multiple confinement n.

Some comment regarding feedsupply optimisation.

    - In the big model we hypothesis that more FVP will be more valuable than more n.
    - It may be sensible to do some manual smoothing of the fs. This can be done in the excel fs which is used
      as the starting point.
    - There some fluctuations in profit as the model finds the optimum feedsupply. Part of this is due to the condensing
      because an animal may choose one feedsupply and then get distributed to a different feedsupply. However if the
      pattern becomes the std pattern then it transfers 1:1 which means an animal may not get its optimal feedsupply
      in the following period. Over a couple of runs it seems to sort its self out. Alternative is to REV all the
      condensed variables but this is not the neatest solution and only works if the condensed weights are sensible.

'''
'''
Additional notes:
- if running n1 with pkl create and use = True and nut spread != 0 then the feed supply will change each iteration 
  because the pkl fs is read in before nut adjustment but stored for post calculating after. Therefore each iteration 
  the fs will change by an amount specified by nut spread.
- cant optimise feeds using multiprocessing because the same pkl file is accessed and written too.
'''

import numpy as np
import pickle as pkl
import os.path

from . import PropertyInputs as pinp
from . import UniversalInputs as uinp
from . import StructuralInputs as sinp
from . import StockFunctions as sfun
from . import Functions as fun
from . import FeedsupplyFunctions as fsfun
from . import SeasonalFunctions as zfun
from . import Sensitivity as sen
from . import relativeFile

na=np.newaxis


#todo supp feeding in confinement incurs the same costs as paddock feeding. this should be changed. it should also incur some capital cost.
# it should also have less costs and time for feeding

def f1_stock_fs(cr_sire,cr_dams,cr_offs,cu0_sire,cu0_dams,cu0_offs,a_p6_pa1e1b1nwzida0e0b0xyg,
                 period_between_weanprejoin_pa1e1b1nwzida0e0b0xyg1,
                 scan_management_pa1e1b1nwzida0e0b0xyg1, gbal_management_pa1e1b1nwzida0e0b0xyg1, wean_management_pa1e1b1nwzida0e0b0xyg1,
                 a_n_pa1e1b1nwzida0e0b0xyg1, a_n_pa1e1b1nwzida0e0b0xyg3, a_t_tpg1, i_g3_inc, mask_p_offs_p, len_p, pkl_fs_info, pkl_fs):

    #########
    #inputs #
    #########
    ##masks required for initialising arrays
    mask_sire_inc_g0 = np.any(sinp.stock['i_mask_g0g3'] * i_g3_inc, axis =1)
    mask_dams_inc_g1 = np.any(sinp.stock['i_mask_g1g3'] * i_g3_inc, axis =1)
    mask_yatf_inc_g2 = np.any(sinp.stock['i_mask_g2g3'] * i_g3_inc, axis =1)
    mask_offs_inc_g3 = np.any(sinp.stock['i_mask_g3g3'] * i_g3_inc, axis =1)

    ##pos
    a0_pos = sinp.stock['i_a0_pos']
    a1_pos = sinp.stock['i_a1_pos']
    b0_pos = sinp.stock['i_b0_pos']
    b1_pos = sinp.stock['i_b1_pos']
    d_pos = sinp.stock['i_d_pos']
    e1_pos = sinp.stock['i_e1_pos']
    g_pos = -1
    i_pos = sinp.stock['i_i_pos']
    n_pos = sinp.stock['i_n_pos']
    p_pos = sinp.stock['i_p_pos']
    w_pos = sinp.stock['i_w_pos']
    x_pos = sinp.stock['i_x_pos']
    z_pos = sinp.stock['i_z_pos']

    ##len
    len_a0 = np.count_nonzero(pinp.sheep['i_mask_a'])
    len_a1 = np.count_nonzero(pinp.sheep['i_mask_a'])


    ##nut
    n_fs_sire = sinp.structuralsa['i_n0_len']
    n_fs_dams = sinp.structuralsa['i_n1_len']
    n_fs_offs = sinp.structuralsa['i_n3_len']

    ##pasture params
    cu3 = uinp.pastparameters['i_cu3_c4'][...,pinp.sheep['i_pasture_type']].astype(float)#have to convert from object to float so it doesn't chuck error in np.exp (np.exp can't handle object arrays)
    cu4 = uinp.pastparameters['i_cu4_c4'][...,pinp.sheep['i_pasture_type']].astype(float)#have to convert from object to float so it doesn't chuck error in np.exp (np.exp can't handle object arrays)

    ##legume proportion in each period
    legume_p6a1e1b1nwzida0e0b0xyg = fun.f_expand(pinp.sheep['i_legume_p6z'], z_pos, move=True, source=0, dest=-1,
                                                 left_pos2=p_pos, right_pos2=z_pos)
    ##Height ratio scalar for the region
    hr_scalar = pinp.sheep['i_hr_scalar']
    ##estimated foo and dmd for the feed periods (p6) periods
    paststd_foo_p6a1e1b1j0wzida0e0b0xyg = fun.f_expand(pinp.sheep['i_paststd_foo_zp6j0'],z_pos,move=True,source=0,
                                                       dest=2, left_pos2=n_pos,right_pos2=z_pos,left_pos3=p_pos,
                                                       right_pos3=n_pos)
    pasture_stage_p6a1e1b1j0wzida0e0b0xyg = fun.f_expand(pinp.sheep['i_pasture_stage_p6z'], z_pos, move=True, source=0,
                                                         dest=-1, left_pos2=p_pos, right_pos2=z_pos)  # z is treated in next step
    ##foo corrected to GrazPlan units and estimated height - the z axis is also treated in this step
    paststd_foo_p6a1e1b1j0wzida0e0b0xyg0, paststd_hf_p6a1e1b1j0wzida0e0b0xyg0 = fsfun.f_foo_convert(cu3, cu4,
                                                                                     paststd_foo_p6a1e1b1j0wzida0e0b0xyg,
                                                                                     pasture_stage_p6a1e1b1j0wzida0e0b0xyg,
                                                                                     legume_p6a1e1b1nwzida0e0b0xyg,
                                                                                     hr_scalar, cr_sire,
                                                                                     z_pos=sinp.stock['i_z_pos'], treat_z=True)
    paststd_foo_p6a1e1b1j0wzida0e0b0xyg1, paststd_hf_p6a1e1b1j0wzida0e0b0xyg1 = fsfun.f_foo_convert(cu3, cu4,
                                                                                     paststd_foo_p6a1e1b1j0wzida0e0b0xyg,
                                                                                     pasture_stage_p6a1e1b1j0wzida0e0b0xyg,
                                                                                     legume_p6a1e1b1nwzida0e0b0xyg,
                                                                                     hr_scalar, cr_dams,
                                                                                     z_pos=sinp.stock['i_z_pos'], treat_z=True)
    paststd_foo_p6a1e1b1j0wzida0e0b0xyg3, paststd_hf_p6a1e1b1j0wzida0e0b0xyg3 = fsfun.f_foo_convert(cu3, cu4,
                                                                                     paststd_foo_p6a1e1b1j0wzida0e0b0xyg,
                                                                                     pasture_stage_p6a1e1b1j0wzida0e0b0xyg,
                                                                                     legume_p6a1e1b1nwzida0e0b0xyg,
                                                                                     hr_scalar, cr_offs,
                                                                                     z_pos=sinp.stock['i_z_pos'], treat_z=True)
    ##treat z axis (have to do it after adjusting foo)
    legume_p6a1e1b1nwzida0e0b0xyg = zfun.f_seasonal_inp(legume_p6a1e1b1nwzida0e0b0xyg,numpy=True,axis=z_pos)
    ##dmd
    paststd_dmd_p6a1e1b1j0wzida0e0b0xyg = fun.f_expand(pinp.sheep['i_paststd_dmd_zp6j0'],z_pos,move=True,source=0,
                                                       dest=2, left_pos2=n_pos,right_pos2=z_pos,left_pos3=p_pos,
                                                       right_pos3=n_pos)
    paststd_dmd_p6a1e1b1j0wzida0e0b0xyg = zfun.f_seasonal_inp(paststd_dmd_p6a1e1b1j0wzida0e0b0xyg,numpy=True,axis=z_pos)
    ##expected supplementary feeding level in each period (propn of intake)
    std_supp_p6a1e1b1nwzida0e0b0xyg = zfun.f_seasonal_inp(fun.f_expand(pinp.sheep['i_supplement_zp6'],swap=True,left_pos=z_pos, left_pos2=p_pos,
                                                                       right_pos2=z_pos)
                                                           ,numpy=True,axis=z_pos)


    ###############################
    # relate NV to FOO, DMD & Supp #
    ###############################
    '''
    AFO feedsupply inputs used to generate livestock are nutritive values however the generator needs information
    about the feed consumed e.g. FOO and DMD because these factors impact things like energy required (e.g. less feed means
    more walking). The function creates a relationship between NV and feed components.

    yatf don't require this function call since they just get the same fs as dams.
    '''
    # todo zf needs to be added to this function when cattle are added.
    ##sires
    nv_p6a1e1b1j1wzida0e0b0xyg0,foo_p6a1e1b1j1wzida0e0b0xyg0,dmd_p6a1e1b1j1wzida0e0b0xyg0,supp_p6a1e1b1j1wzida0e0b0xyg0 = \
        sfun.f1_nv_components(paststd_foo_p6a1e1b1j0wzida0e0b0xyg0,paststd_dmd_p6a1e1b1j0wzida0e0b0xyg,
                              paststd_hf_p6a1e1b1j0wzida0e0b0xyg0,std_supp_p6a1e1b1nwzida0e0b0xyg,
                              legume_p6a1e1b1nwzida0e0b0xyg,cr_sire,cu0_sire)

    ##dams
    nv_p6a1e1b1j1wzida0e0b0xyg1,foo_p6a1e1b1j1wzida0e0b0xyg1,dmd_p6a1e1b1j1wzida0e0b0xyg1,supp_p6a1e1b1j1wzida0e0b0xyg1 = \
        sfun.f1_nv_components(paststd_foo_p6a1e1b1j0wzida0e0b0xyg1,paststd_dmd_p6a1e1b1j0wzida0e0b0xyg,
                              paststd_hf_p6a1e1b1j0wzida0e0b0xyg1,std_supp_p6a1e1b1nwzida0e0b0xyg,
                              legume_p6a1e1b1nwzida0e0b0xyg,cr_dams,cu0_dams)

    ##offs
    nv_p6a1e1b1j1wzida0e0b0xyg3,foo_p6a1e1b1j1wzida0e0b0xyg3,dmd_p6a1e1b1j1wzida0e0b0xyg3,supp_p6a1e1b1j1wzida0e0b0xyg3 = \
        sfun.f1_nv_components(paststd_foo_p6a1e1b1j0wzida0e0b0xyg3,paststd_dmd_p6a1e1b1j0wzida0e0b0xyg,
                              paststd_hf_p6a1e1b1j0wzida0e0b0xyg3,std_supp_p6a1e1b1nwzida0e0b0xyg,
                              legume_p6a1e1b1nwzida0e0b0xyg,cr_offs,cu0_offs)


    ############################
    ### feed supply calcs      #
    ############################
    ##r1 & r2 are the axes for the inputs that are the options of different feed supply.
    # r1 is the choices for the full year feed supply for the undifferentiated animal.
    # r2 is the adjustment for different classes or different management.

    ##feedsupply selection inputs
    ###feedsupply option selected - keep the z axis here and then handle the z axis after the feedsupply is calculated
    a_r_zida0e0b0xyg0 = fun.f_expand(pinp.sheep['ia_r1_zig0'], i_pos, right_pos=g_pos, swap=True, condition=mask_sire_inc_g0, axis=g_pos,
                                     condition2=pinp.sheep['i_masksire_i'], axis2=i_pos).astype(int)
    a_r_zida0e0b0xyg1 = fun.f_expand(pinp.sheep['ia_r1_zig1'], i_pos, right_pos=g_pos, swap=True, condition=mask_dams_inc_g1, axis=g_pos,
                                     condition2=pinp.sheep['i_mask_i'], axis2=i_pos).astype(int)
    a_r_zida0e0b0xyg3 = fun.f_expand(pinp.sheep['ia_r1_zig3'], i_pos, right_pos=g_pos, swap=True, condition=mask_offs_inc_g3, axis=g_pos,
                                     condition2=pinp.sheep['i_mask_i'], axis2=i_pos).astype(int)
    ###feed adjustment for dams
    a_r2_k0e1b1nwzida0e0b0xyg1 = fun.f_expand(pinp.sheep['ia_r2_ik0g1'], i_pos, right_pos=g_pos, swap=True, left_pos2=a1_pos
                                             , right_pos2=i_pos, condition=mask_dams_inc_g1, axis=g_pos,
                                             condition2=pinp.sheep['i_mask_i'], axis2=i_pos)
    a_r2_k1b1nwzida0e0b0xyg1 = fun.f_expand(pinp.sheep['ia_r2_ik1g1'], i_pos, right_pos=g_pos, swap=True, left_pos2=e1_pos
                                           , right_pos2=i_pos, condition=mask_dams_inc_g1, axis=g_pos,
                                           condition2=pinp.sheep['i_mask_i'], axis2=i_pos)
    a_r2_spk0k1k2nwzida0e0b0xyg1 = fun.f_expand(pinp.sheep['ia_r2_isk2g1'], i_pos, right_pos=g_pos, left_pos2=b1_pos, right_pos2=i_pos
                                               , left_pos3=p_pos-1, right_pos3=b1_pos, condition=mask_dams_inc_g1, axis=g_pos,
                                               condition2=pinp.sheep['i_mask_i'], axis2=i_pos, move=True, source=0, dest=2)  #add axes between g and i and i and b1
    ###feed adjustment for offs
    a_r2_idk0e0b0xyg3 = fun.f_expand(pinp.sheep['ia_r2_ik0g3'], a0_pos, right_pos=g_pos, left_pos2=i_pos, right_pos2=a0_pos,
                                    condition=mask_offs_inc_g3, axis=g_pos, condition2=pinp.sheep['i_mask_i'], axis2=i_pos)
    a_r2_ik3a0e0b0xyg3 = fun.f_expand(pinp.sheep['ia_r2_ik3g3'], d_pos, right_pos=g_pos, condition=mask_offs_inc_g3, axis=g_pos, condition2=pinp.sheep['i_mask_i'], axis2=i_pos)
    a_r2_ida0e0k4xyg3 = fun.f_expand(pinp.sheep['ia_r2_ik4g3'], b0_pos, right_pos=g_pos, left_pos2=i_pos, right_pos2=b0_pos,
                                    condition=mask_offs_inc_g3, axis=g_pos, condition2=pinp.sheep['i_mask_i'], axis2=i_pos)  #add axis between g and b0 and b0 and i
    a_r2_ida0e0b0k5yg3 = fun.f_expand(pinp.sheep['ia_r2_ik5g3'], x_pos, right_pos=g_pos, left_pos2=i_pos, right_pos2=x_pos,
                                     condition=mask_offs_inc_g3, axis=g_pos, condition2=pinp.sheep['i_mask_i'], axis2=i_pos)  #add axis between g and b0 and b0 and i

    ##std feed options
    feedsupply_options_r1j2p = pinp.feedsupply['i_feedsupply_options_r1j2p'][...,0:len_p].astype(float) #slice off extra p periods so it is the same length as the sim periods
    ##confinement - True/False on confinement feeding. This controls which generator periods confinement feeding occurs.
    ## this input is required so that confinement can be included in n1 model, without forcing the animal into confinement for the whole year.
    ## This means you can have a given level of NV and it can be either in the paddock or in confinement.
    ## This input works in conjunction with i_confinement_n (see in the feed supply section)
    confinement_options_r1p6z = pinp.feedsupply['i_confinement_options_r1p6z'].astype(float)
    ##feed supply adjustment
    feedsupply_adj_options_r2p = pinp.feedsupply['i_feedsupply_adj_options_r2p'][:,0:len_p].astype(float) #slice off extra p periods so it is the same length as the sim periods
    ##an association between the k2 cluster (feed adjustment) and reproductive management (scanning, gbal & weaning).
    a_k2_mlsb1 = sinp.stock['ia_k2_mlsb1']


    ##1a) compile the standard pattern from the inputs and handle the z axis (need to apply z treatment here because a_r_zida0e0b0xyg0 didn't get the season treatment)
    ###sire
    t_feedsupply_pj2zida0e0b0xyg0 = np.moveaxis(np.moveaxis(feedsupply_options_r1j2p[a_r_zida0e0b0xyg0],-1,0),-1,1) #had to rollaxis twice once for p and once for j2 (couldn't find a way to do both at the same time)
    t_feedsupply_pj2zida0e0b0xyg0 = zfun.f_seasonal_inp(t_feedsupply_pj2zida0e0b0xyg0,numpy=True,axis=z_pos)
    t_feedsupply_pa1e1b1j2wzida0e0b0xyg0 = fun.f_expand(t_feedsupply_pj2zida0e0b0xyg0, left_pos=n_pos, right_pos=z_pos, left_pos2=p_pos,right_pos2=n_pos) #add  a1,e1,b1,w axis. Note n and j are the same thing (as far a position goes)

    ###dams
    t_feedsupply_pj2zida0e0b0xyg1 = np.moveaxis(np.moveaxis(feedsupply_options_r1j2p[a_r_zida0e0b0xyg1],-1,0),-1,1) #had to rollaxis twice once for p and once for j2 (couldn't find a way to do both at the same time)
    t_feedsupply_pj2zida0e0b0xyg1 = zfun.f_seasonal_inp(t_feedsupply_pj2zida0e0b0xyg1,numpy=True,axis=z_pos)
    t_feedsupply_pa1e1b1j2wzida0e0b0xyg1 = fun.f_expand(t_feedsupply_pj2zida0e0b0xyg1, left_pos=n_pos, right_pos=z_pos, left_pos2=p_pos,right_pos2=n_pos) #add  a1,e1,b1,w axis. Note n and j are the same thing (as far a position goes)

    ###offs
    t_feedsupply_pj2zida0e0b0xyg3 = np.moveaxis(np.moveaxis(feedsupply_options_r1j2p[a_r_zida0e0b0xyg3],-1,0),-1,1) #had to rollaxis twice once for p and once for j2 (couldn't find a way to do both at the same time)
    t_feedsupply_pj2zida0e0b0xyg3 = zfun.f_seasonal_inp(t_feedsupply_pj2zida0e0b0xyg3,numpy=True,axis=z_pos)
    t_feedsupply_pa1e1b1j2wzida0e0b0xyg3 = fun.f_expand(t_feedsupply_pj2zida0e0b0xyg3, left_pos=n_pos, right_pos=z_pos, left_pos2=p_pos,right_pos2=n_pos, condition=mask_p_offs_p, axis=0) #add  a1,e1,b1,w axis. Note n and j are the same thing (as far a position goes), mask p axis for offs

    ##1b) select confinement options for each animal group and handle the z axis (need to apply z treatment here because a_r_zida0e0b0xyg0 didn't get the season treatment)
    ## using advanced indexing (could have also used np.take_along_axis but that requires getting all the arrays the same shape)
    t_index_zida0e0b0xyg = fun.f_expand(np.arange(confinement_options_r1p6z.shape[-1]),z_pos) #length of z before being masked. needs to broadcast with a_r_zida0e0b0xyg so advanced indexing works
    ###sire
    t_confinement_p6zida0e0b0xyg0 = np.moveaxis(confinement_options_r1p6z[a_r_zida0e0b0xyg0,:,t_index_zida0e0b0xyg],-1,0) #move p to front
    t_confinement_p6zida0e0b0xyg0 = zfun.f_seasonal_inp(t_confinement_p6zida0e0b0xyg0,numpy=True,axis=z_pos)
    t_confinement_p6a1e1b1nwzida0e0b0xyg0 = fun.f_expand(t_confinement_p6zida0e0b0xyg0, left_pos=p_pos, right_pos=z_pos) #add  a1,e1,b1,w,n axis.
    t_confinement_pa1e1b1nwzida0e0b0xyg0 = np.take_along_axis(t_confinement_p6a1e1b1nwzida0e0b0xyg0,a_p6_pa1e1b1nwzida0e0b0xyg,0)

    ###dams
    t_confinement_p6zida0e0b0xyg1 = np.moveaxis(confinement_options_r1p6z[a_r_zida0e0b0xyg1,:,t_index_zida0e0b0xyg],-1,0) #move p to front
    t_confinement_p6zida0e0b0xyg1 = zfun.f_seasonal_inp(t_confinement_p6zida0e0b0xyg1,numpy=True,axis=z_pos)
    t_confinement_p6a1e1b1nwzida0e0b0xyg1 = fun.f_expand(t_confinement_p6zida0e0b0xyg1, left_pos=p_pos, right_pos=z_pos) #add  a1,e1,b1,w,n axis.
    t_confinement_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(t_confinement_p6a1e1b1nwzida0e0b0xyg1,a_p6_pa1e1b1nwzida0e0b0xyg,0)

    ###offs
    t_confinement_p6zida0e0b0xyg3 = np.moveaxis(confinement_options_r1p6z[a_r_zida0e0b0xyg3,:,t_index_zida0e0b0xyg],-1,0)#move p to front
    t_confinement_p6zida0e0b0xyg3 = zfun.f_seasonal_inp(t_confinement_p6zida0e0b0xyg3,numpy=True,axis=z_pos)
    t_confinement_p6a1e1b1nwzida0e0b0xyg3 = fun.f_expand(t_confinement_p6zida0e0b0xyg3, left_pos=p_pos, right_pos=z_pos, axis=0) #add  a1,e1,b1,w,n axis.
    t_confinement_pa1e1b1nwzida0e0b0xyg3 = np.take_along_axis(t_confinement_p6a1e1b1nwzida0e0b0xyg3,a_p6_pa1e1b1nwzida0e0b0xyg[mask_p_offs_p],0)

    ##2) calculate the feedsupply adjustment OPTION for each sheep class
    ###a)wean
    a_k0_pa1e1b1nwzida0e0b0xyg1 = period_between_weanprejoin_pa1e1b1nwzida0e0b0xyg1 * pinp.sheep['i_dam_wean_diffman'] * fun.f_expand(np.arange(len_a1)+1, a1_pos) #len_a+1 because that is the association between k0 and a1
    a_r2_wean_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(a_r2_k0e1b1nwzida0e0b0xyg1[na,...], a_k0_pa1e1b1nwzida0e0b0xyg1, a1_pos)

    ###b)b.	Dams Cluster k1 – oestrus cycle (e1): The association required is
    #^Have decided to drop this out of version 1. Will require multiple nutrition patterns in order to test value of scanning for foetal age
    # a_r2_oestrus_pa1e1b1nwzida0e0b0xyg1 =

    ###c)lsln
    ####a_k2_mlsb1 is the k2 input cluster for each b1 slice (LSLN) with different management options.
    #### In this step we slice a_k2_mlsb1 for the selected management in each period.
    #####remove the singleton b1 axis from the association arrays because a populated b1 axis comes from a_k2_mlsb1
    a_k2_pa1e1b1nwzida0e0b0xyg1 = np.rollaxis(a_k2_mlsb1[wean_management_pa1e1b1nwzida0e0b0xyg1[:,:,:,0,...]
                                                         , gbal_management_pa1e1b1nwzida0e0b0xyg1[:,:,:,0,...]
                                                         , scan_management_pa1e1b1nwzida0e0b0xyg1[:,:,:,0,...], ...],-1,3)
    ####a_r2_spk0k1k2nwzida0e0b0xyg1 (k2 active) is the feedsupply adjustment option for each k2 input cluster for each scanning options.
    ####The scan axis is required because the feedsupply for a cluster can vary based on how the other classes are clustered
    #### e.g. the optimum feedsupply prior to scanning may change depending on whether singles and twins are identified
    #todo the above comment is correct however, it is not represented in the inputs of the model. Undiff/mated is always 0 regardless of the scanning level
    #####take along the scan axis then remove the singleton scan axis with [0]
    a_r2_pk0k1k2nwzida0e0b0xyg1 = np.take_along_axis(a_r2_spk0k1k2nwzida0e0b0xyg1, scan_management_pa1e1b1nwzida0e0b0xyg1[na,...], axis=0)[0]
    ####select feedsupply adjustment option for each b slice based on the 'k2 input cluster' association.
    a_r2_lsln_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(a_r2_pk0k1k2nwzida0e0b0xyg1, a_k2_pa1e1b1nwzida0e0b0xyg1, b1_pos)

    ###d) todo come back to offs (remember gender needs to be masked)
    # t_fs_agedam_pj2zida0e0b0xg3 = t_fs_agedam_pj2zik3k0k4k5g3
    # t_fs_ageweaned_pj2zida0e0b0xg3 = t_fs_ageweaned_pj2zik3k0k4k5g3
    a_k0_a0e0b0xyg3 = pinp.sheep['i_off_wean_diffman'] * fun.f_expand(np.arange(len_a0)+1, a0_pos) #len_a+1 because that is the association between k0 and a1
    a_r2_wean_ida0e0b0xyg3 = np.take_along_axis(a_r2_idk0e0b0xyg3, a_k0_a0e0b0xyg3[na,na,...], a0_pos)
    a_r2_wean_pa1e1b1nwzida0e0b0xyg3 = fun.f_expand(a_r2_wean_ida0e0b0xyg3, left_pos=p_pos-1, right_pos=i_pos)  #p_pos-1 because require a p axis and all are new axes.
    # t_fs_btrt_pj2zida0e0b0xg3 = t_fs_btrt_pj2zik3k0k4k5g3
    # t_fs_gender_pj2zida0e0b0xg3 = t_fs_gender_pj2zik3k0k4k5g3


    ##3) calculate the feedsupply adjustment for each sheep class
    feedsupply_adj_options_r2pa1e1b1nwzida0e0b0xyg1 = fun.f_expand(feedsupply_adj_options_r2p, p_pos) #add other axis as singleton
    feedsupply_adj_options_r2pa1e1b1nwzida0e0b0xyg3 = fun.f_expand(feedsupply_adj_options_r2p, p_pos
                                                                   , condition=mask_p_offs_p, axis=p_pos)
    ###a)wean (take along the r2 axis and then remove the singleton axis with [0])
    t_fs_ageweaned_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(feedsupply_adj_options_r2pa1e1b1nwzida0e0b0xyg1
                                                                , a_r2_wean_pa1e1b1nwzida0e0b0xyg1[na,...], axis=0)[0]
    ###b)oestrus (take along the r2 axis and then remove the singleton axis with [0])
    # t_fs_cycle_pa1e1b1j2wzida0e0b0xyg1 = np.take_along_axis(feedsupply_adj_options_r2pa1e1b1nwzida0e0b0xyg1
    #                                                         , a_r2_oestrus_pa1e1b1nwzida0e0b0xyg1[na,...], axis=0)[0]
    ###c)lsln (take along the r2 axis and then remove the singleton axis with [0])
    t_fs_lsln_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(feedsupply_adj_options_r2pa1e1b1nwzida0e0b0xyg1
                                                           , a_r2_lsln_pa1e1b1nwzida0e0b0xyg1[na,...], axis=0)[0]

    ###d)agedam for offspring
    t_fs_agedam_pa1e1b1nwzik3a0e0b0xyg3 = 0
    ###e)wean age for offspring
    t_fs_ageweaned_pa1e1b1nwzidk0e0b0xyg3 = np.take_along_axis(feedsupply_adj_options_r2pa1e1b1nwzida0e0b0xyg3
                                                                , a_r2_wean_pa1e1b1nwzida0e0b0xyg3[na,...], axis=0)[0]
    ###f) btrt for offspring
    t_fs_btrt_pa1e1b1nwzida0e0k4xyg3 = 0
    ###g) gender for offspring
    t_fs_gender_pa1e1b1nwzida0e0b0k5yg3 = 0

    ##4a) update fs and confinement info with pkl if desired
    ### It should be possible to use a BBB pkl_fs with a BBT flock if the BBT flock is generating without the t axis.
    ###Generating without t means that the t axis of pkl_fs is reduced to a singleton, then t & g will broadcast.
    if sinp.structuralsa['i_fs_use_pkl']:
        ###update the feedsupply with the pkl fs
        t_feedsupply_tpa1e1b1j2wzida0e0b0xyg0 = pkl_fs['fs']['sire']
        t_feedsupply_stpa1e1b1j2wzida0e0b0xyg1 = pkl_fs['fs']['dams']
        t_feedsupply_stpa1e1b1j2wzida0e0b0xyg3 = pkl_fs['fs']['offs']

        ###confinement info - only use if n=1 and fs_use_pkl (above) is true otherwise fs optimisation could get stuck at local optimum
        t_confinement_tpa1e1b1nwzida0e0b0xyg0 = fun.f_update(t_confinement_pa1e1b1nwzida0e0b0xyg0[na], pkl_fs['confinement']['sire'], n_fs_sire==1)
        t_confinement_stpa1e1b1nwzida0e0b0xyg1 = fun.f_update(t_confinement_pa1e1b1nwzida0e0b0xyg1[na], pkl_fs['confinement']['dams'], n_fs_dams==1)
        t_confinement_stpa1e1b1nwzida0e0b0xyg3 = fun.f_update(t_confinement_pa1e1b1nwzida0e0b0xyg3[na], pkl_fs['confinement']['offs'], n_fs_offs==1)

        ###slice t axis if t is not being included in the looping part of sgen
        ### take the 'retain' t slice (slice so that singleton axis remains)
        if not sinp.structuralsa['i_generate_with_t']:
            t_confinement_stpa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(t_confinement_stpa1e1b1nwzida0e0b0xyg1, a_t_tpg1[na], axis=p_pos - 1)
            t_confinement_stpa1e1b1nwzida0e0b0xyg3 = t_confinement_stpa1e1b1nwzida0e0b0xyg3[:,0:1]
            t_feedsupply_stpa1e1b1j2wzida0e0b0xyg1 = np.take_along_axis(t_feedsupply_stpa1e1b1j2wzida0e0b0xyg1, a_t_tpg1[na], axis=p_pos - 1)
            t_feedsupply_stpa1e1b1j2wzida0e0b0xyg3 = t_feedsupply_stpa1e1b1j2wzida0e0b0xyg3[:,0:1]

        #todo a_k2_pa1e1b1nwzida0e0b0xyg1 is not right for this purpose. We need to build a new a_b1_p... association.
        # maybe  ia_ppk2_vlsb1 (although this will need a fair bit of work to get into the right structure)
        # off might also need something similar to below for their fs using ia_ppk5g3_lsb0
        ###for dams adjust fs for scanning/gbal management. This handles user error if optimal fs is generated with say scan 3 but then used for a scan 2
        # t_confinement_stpa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(t_confinement_stpa1e1b1nwzida0e0b0xyg1, a_k2_pa1e1b1nwzida0e0b0xyg1[na,na,...], b1_pos)
        # t_feedsupply_stpa1e1b1j2wzida0e0b0xyg1 = np.take_along_axis(t_feedsupply_stpa1e1b1j2wzida0e0b0xyg1, a_k2_pa1e1b1nwzida0e0b0xyg1[na,na,...], b1_pos)

    ###still need to add singleton s&t axis
    else:
        t_confinement_tpa1e1b1nwzida0e0b0xyg0 = t_confinement_pa1e1b1nwzida0e0b0xyg0[na]
        t_confinement_stpa1e1b1nwzida0e0b0xyg1 = t_confinement_pa1e1b1nwzida0e0b0xyg1[na,na]
        t_confinement_stpa1e1b1nwzida0e0b0xyg3 = t_confinement_pa1e1b1nwzida0e0b0xyg3[na,na]
        t_feedsupply_tpa1e1b1j2wzida0e0b0xyg0 = t_feedsupply_pa1e1b1j2wzida0e0b0xyg0[na]
        t_feedsupply_stpa1e1b1j2wzida0e0b0xyg1 = t_feedsupply_pa1e1b1j2wzida0e0b0xyg1[na,na]
        t_feedsupply_stpa1e1b1j2wzida0e0b0xyg3 = t_feedsupply_pa1e1b1j2wzida0e0b0xyg3[na,na]

    ##4b) apply confinement SA - note this will overwrite pkl confinement
    ## This allows the user to adjust confinement  by p axis because the pinp input is only p6.
    sa_dams_confinement_pg1 = fun.f_expand(sen.sav['dams_confinement_P'][0:len_p],p_pos) #slice P axis and expand axes
    t_confinement_stpa1e1b1nwzida0e0b0xyg1 = fun.f_sa(t_confinement_stpa1e1b1nwzida0e0b0xyg1, sa_dams_confinement_pg1, 5)
    sa_offs_confinement_pg3 = fun.f_expand(sen.sav['offs_confinement_P'][0:mask_p_offs_p.sum()],p_pos) #slice P axis and expand axes
    t_confinement_stpa1e1b1nwzida0e0b0xyg3 = fun.f_sa(t_confinement_stpa1e1b1nwzida0e0b0xyg3, sa_offs_confinement_pg3, 5)

    ##4c) add adjustment to std pattern if it is specified (mostly won't be included if using pkl_fs)
    ## adjustment is only calculated for dams
    if sinp.structuralsa['i_r2adjust_inc']:
        ##the adjustment is broadcast across j2 (the standard, minimum and maximum)
        ##feedsupply is clipped in step 7 to ensure it is within a feasible range.
        t_feedsupply_stpa1e1b1j2wzida0e0b0xyg1 = (t_feedsupply_stpa1e1b1j2wzida0e0b0xyg1 + t_fs_ageweaned_pa1e1b1nwzida0e0b0xyg1
                                                  + t_fs_lsln_pa1e1b1nwzida0e0b0xyg1) #can't use += for some reason
        t_feedsupply_stpa1e1b1j2wzida0e0b0xyg3 = (t_feedsupply_stpa1e1b1j2wzida0e0b0xyg3 + t_fs_agedam_pa1e1b1nwzik3a0e0b0xyg3
                                                  + t_fs_ageweaned_pa1e1b1nwzidk0e0b0xyg3 + t_fs_btrt_pa1e1b1nwzida0e0k4xyg3
                                                  + t_fs_gender_pa1e1b1nwzida0e0b0k5yg3)

    ##5)Convert the ‘j2’ axis to an ‘n’ axis using the nut_spread inputs.
    ## activate n axis for confinement control (controls if a nutrition pattern is in confinement - note
    ## if i_confinement_n? is set to True the generator periods confinement occurs is controlled by i_confinement_options_r1p6z.
    feedsupply_std_tpa1e1b1nwzida0e0b0xyg0, confinement_std_tpa1e1b1nwzida0e0b0xyg0, bool_confinement_g0_n =  \
        f1_j2_to_n(t_feedsupply_tpa1e1b1j2wzida0e0b0xyg0, t_confinement_tpa1e1b1nwzida0e0b0xyg0,
                   a_p6_pa1e1b1nwzida0e0b0xyg, sinp.structuralsa['i_nut_spread_n0'], sinp.structuralsa['i_confinement_n0'],
                   n_fs_sire)

    feedsupply_std_stpa1e1b1nwzida0e0b0xyg1, confinement_std_stpa1e1b1nwzida0e0b0xyg1, bool_confinement_g1_n = \
        f1_j2_to_n(t_feedsupply_stpa1e1b1j2wzida0e0b0xyg1, t_confinement_stpa1e1b1nwzida0e0b0xyg1,
                   a_p6_pa1e1b1nwzida0e0b0xyg, sinp.structuralsa['i_nut_spread_n1'], sinp.structuralsa['i_confinement_n1'],
                   n_fs_dams)
    feedsupply_std_stpa1e1b1nwzida0e0b0xyg3, confinement_std_stpa1e1b1nwzida0e0b0xyg3, bool_confinement_g3_n = \
        f1_j2_to_n(t_feedsupply_stpa1e1b1j2wzida0e0b0xyg3, t_confinement_stpa1e1b1nwzida0e0b0xyg3,
                   a_p6_pa1e1b1nwzida0e0b0xyg[mask_p_offs_p], sinp.structuralsa['i_nut_spread_n3'], sinp.structuralsa['i_confinement_n3'],
                   n_fs_offs)

    ##6) expand feedsupply for all w slices - see google doc for more info
    ###convert feedsupply from having an active n axis to active w axis by applying association
    ### each slice of w has a combination of nutrition levels during the nutrition cycle and that is specified in the association a_n_pw
    ### if generating from pkl then there is a starting weight axis which has to be reshaped to the full w.
    ###Note: if start w len is different in the current trial compared to the pkl fs then the pkl feedsupply is sliced for start weight [0]. Thus all start weights get the same FS.
    if sinp.structuralsa['i_fs_use_pkl'] and (sinp.structuralsa['i_w_start_len1']==feedsupply_std_stpa1e1b1nwzida0e0b0xyg1.shape[0]):
        ###dams
        len_w1 = a_n_pa1e1b1nwzida0e0b0xyg1.shape[w_pos]
        l_nut1 = int(len_w1/sinp.structuralsa['i_w_start_len1'])#number of nutrition patterns
        feedsupplyw_stpa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(feedsupply_std_stpa1e1b1nwzida0e0b0xyg1, a_n_pa1e1b1nwzida0e0b0xyg1[na,na,:,:,:,:,:,0:l_nut1,...], axis=n_pos)
        feedsupplyw_tpa1e1b1nwzida0e0b0xyg1 = fun.f_merge_axis(feedsupplyw_stpa1e1b1nwzida0e0b0xyg1, 0, w_pos)
        confinementw_stpa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(confinement_std_stpa1e1b1nwzida0e0b0xyg1, a_n_pa1e1b1nwzida0e0b0xyg1[na,na,:,:,:,:,:,0:l_nut1,...], axis=n_pos)
        confinementw_tpa1e1b1nwzida0e0b0xyg1 = fun.f_merge_axis(confinementw_stpa1e1b1nwzida0e0b0xyg1, 0, w_pos)
    else:
        feedsupplyw_tpa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(feedsupply_std_stpa1e1b1nwzida0e0b0xyg1[0], a_n_pa1e1b1nwzida0e0b0xyg1[na], axis=n_pos) #slice off the singleton s axis (this also handles cases where pkl fs has more starting weights than the current run)
        confinementw_tpa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(confinement_std_stpa1e1b1nwzida0e0b0xyg1[0], a_n_pa1e1b1nwzida0e0b0xyg1[na], axis=n_pos) #slice off the singleton s axis (this also handles cases where pkl fs has more starting weights than the current run)
    if sinp.structuralsa['i_fs_use_pkl'] and (sinp.structuralsa['i_w_start_len3']==feedsupply_std_stpa1e1b1nwzida0e0b0xyg3.shape[w_pos]):
        ###offs
        len_w3 = a_n_pa1e1b1nwzida0e0b0xyg3.shape[w_pos]
        l_nut3 = int(len_w3/sinp.structuralsa['i_w_start_len3'])#number of nutrition patterns
        feedsupplyw_stpa1e1b1nwzida0e0b0xyg3 = np.take_along_axis(feedsupply_std_stpa1e1b1nwzida0e0b0xyg3, a_n_pa1e1b1nwzida0e0b0xyg3[na,na,:,:,:,:,:,0:l_nut3,...], axis=n_pos)
        feedsupplyw_tpa1e1b1nwzida0e0b0xyg3 = fun.f_merge_axis(feedsupplyw_stpa1e1b1nwzida0e0b0xyg3, 0, w_pos)
        confinementw_stpa1e1b1nwzida0e0b0xyg3 = np.take_along_axis(confinement_std_stpa1e1b1nwzida0e0b0xyg3, a_n_pa1e1b1nwzida0e0b0xyg3[na,na,:,:,:,:,:,0:l_nut3,...], axis=n_pos)
        confinementw_tpa1e1b1nwzida0e0b0xyg3 = fun.f_merge_axis(confinementw_stpa1e1b1nwzida0e0b0xyg3, 0, w_pos)
    else:
        feedsupplyw_tpa1e1b1nwzida0e0b0xyg3 = np.take_along_axis(feedsupply_std_stpa1e1b1nwzida0e0b0xyg3[0], a_n_pa1e1b1nwzida0e0b0xyg3[na], axis=n_pos) #slice off the singleton s axis (this also handles cases where pkl fs has more starting weights than the current run)
        confinementw_tpa1e1b1nwzida0e0b0xyg3 = np.take_along_axis(confinement_std_stpa1e1b1nwzida0e0b0xyg3[0], a_n_pa1e1b1nwzida0e0b0xyg3[na], axis=n_pos) #slice off the singleton s axis (this also handles cases where pkl fs has more starting weights than the current run)
    ###sires
    feedsupplyw_tpa1e1b1nwzida0e0b0xyg0 = feedsupply_std_tpa1e1b1nwzida0e0b0xyg0 #^only one n slice so doesn't need following code yet: np.take_along_axis(feedsupply_std_pa1e1b1nwzida0e0b0xyg0,a_n_pa1e1b1nwzida0e0b0xyg0,axis=n_pos)
    confinementw_tpa1e1b1nwzida0e0b0xyg0 = confinement_std_tpa1e1b1nwzida0e0b0xyg0 #^only one n slice so doesn't need following code yet: np.take_along_axis(feedsupply_std_pa1e1b1nwzida0e0b0xyg0,a_n_pa1e1b1nwzida0e0b0xyg0,axis=n_pos)

    ##store some info required to determine the optimal feedsupply at the end
    ## note: feedsupply is stored after the generator because it could be changed in the target lw loop.
    pkl_fs_info['confinementw_tpa1e1b1nwzida0e0b0xyg0'] = confinementw_tpa1e1b1nwzida0e0b0xyg0
    pkl_fs_info['confinementw_tpa1e1b1nwzida0e0b0xyg1'] = confinementw_tpa1e1b1nwzida0e0b0xyg1
    pkl_fs_info['confinementw_tpa1e1b1nwzida0e0b0xyg3'] = confinementw_tpa1e1b1nwzida0e0b0xyg3
    ###store the inputted fs and confinement so that we can calculate the difference of the min and max j2 slices in comparison to the std slice.
    ### also used to set the feedsupply and confinement for slice that had no animals selected in the optimal solution.
    pkl_fs_info['xl_feedsupply_pa1e1b1j2wzida0e0b0xyg0'] = t_feedsupply_pa1e1b1j2wzida0e0b0xyg0
    pkl_fs_info['xl_feedsupply_pa1e1b1j2wzida0e0b0xyg1'] = t_feedsupply_pa1e1b1j2wzida0e0b0xyg1
    pkl_fs_info['xl_feedsupply_pa1e1b1j2wzida0e0b0xyg3'] = t_feedsupply_pa1e1b1j2wzida0e0b0xyg3
    pkl_fs_info['xl_confinement_pa1e1b1nwzida0e0b0xyg0'] = t_confinement_pa1e1b1nwzida0e0b0xyg0
    pkl_fs_info['xl_confinement_pa1e1b1nwzida0e0b0xyg1'] = t_confinement_pa1e1b1nwzida0e0b0xyg1
    pkl_fs_info['xl_confinement_pa1e1b1nwzida0e0b0xyg3'] = t_confinement_pa1e1b1nwzida0e0b0xyg3

    return legume_p6a1e1b1nwzida0e0b0xyg, bool_confinement_g0_n, bool_confinement_g1_n, bool_confinement_g3_n, \
           nv_p6a1e1b1j1wzida0e0b0xyg0,foo_p6a1e1b1j1wzida0e0b0xyg0,dmd_p6a1e1b1j1wzida0e0b0xyg0,supp_p6a1e1b1j1wzida0e0b0xyg0,\
           nv_p6a1e1b1j1wzida0e0b0xyg1,foo_p6a1e1b1j1wzida0e0b0xyg1,dmd_p6a1e1b1j1wzida0e0b0xyg1,supp_p6a1e1b1j1wzida0e0b0xyg1,\
           nv_p6a1e1b1j1wzida0e0b0xyg3,foo_p6a1e1b1j1wzida0e0b0xyg3,dmd_p6a1e1b1j1wzida0e0b0xyg3,supp_p6a1e1b1j1wzida0e0b0xyg3,\
           feedsupplyw_tpa1e1b1nwzida0e0b0xyg0, feedsupplyw_tpa1e1b1nwzida0e0b0xyg1, feedsupplyw_tpa1e1b1nwzida0e0b0xyg3, \
           confinementw_tpa1e1b1nwzida0e0b0xyg0, confinementw_tpa1e1b1nwzida0e0b0xyg1, confinementw_tpa1e1b1nwzida0e0b0xyg3



def f1_j2_to_n(t_feedsupply_stpa1e1b1j2wzida0e0b0xyg, t_confinement_pa1e1b1nwzida0e0b0xyg,
               a_p6_pa1e1b1nwzida0e0b0xyg, i_nut_spread_n, i_confinement_n, n_fs):
    n_pos = sinp.stock['i_n_pos']
    p_pos = sinp.stock['i_p_pos']
    z_pos = sinp.stock['i_z_pos']
    ### the nut_spread inputs are the proportion of std and min or max feed supply.
    ### Unless nut_spread is greater than 3 in which case the value becomes the actual feed supply
    #todo nutspread >3 doesn't overwrite the value any more.
    ###convert nut_spread inputs to numpy array and cut to the correct length based on number of nutrition options (i_len_n structural input)
    if isinstance(i_nut_spread_n, np.ndarray):
        nut_spread_n = i_nut_spread_n[0:n_fs]
        bool_confinement_n = i_confinement_n[0:n_fs]
    else:
        nut_spread_n = np.array([i_nut_spread_n])[0:n_fs]
        bool_confinement_n = np.array([i_confinement_n])[0:n_fs]


    ###a- create a ‘j2’ by ‘n’ array that is the multipliers that weight each ‘j2’ for that level of ‘n’
    ###the slices of j2 are Std, minimum & maximum respectively
    ###the nut_mult does an array equivalent of feed supply = std + (max - std) * spread (if spread > 0, (min - std) if spread < 0)
    ###the nut_mult step is carried out on NV (MJ of MEI / intake volume required)
    len_j2 = pinp.feedsupply['i_j2_len']
    nut_mult_j2n = np.empty((len_j2,n_fs))
    nut_mult_j2n[0, ...] = 1 - np.abs(nut_spread_n)
    nut_mult_j2n[1, ...] = np.abs(np.minimum(0, nut_spread_n))
    nut_mult_j2n[2, ...] = np.abs(np.maximum(0, nut_spread_n))

    ###b - feedsupply_std with n axis (instead of j axis).
    nut_mult_pk0k1k2j2nwzida0e0b0xyg = np.expand_dims(nut_mult_j2n[na,na,na,na,...], axis = tuple(range(n_pos+1,0))) #expand axis to line up with feedsupply, add axis from g to n and j2 to p
    t_feedsupply_stpa1e1b1j2nwzida0e0b0xyg = np.expand_dims(t_feedsupply_stpa1e1b1j2wzida0e0b0xyg, axis = n_pos) #add n axis
    feedsupply_std_stpa1e1b1nwzida0e0b0xyg = np.sum(t_feedsupply_stpa1e1b1j2nwzida0e0b0xyg * nut_mult_pk0k1k2j2nwzida0e0b0xyg, axis = n_pos-1 ) #sum j axis, minus 1 because n axis was added therefore shifting j2 position (it was originally in the same place). Sum across j2 axis and leave just the n axis

    ###c activate n axis on confinement control
    confinement_std_pa1e1b1nwzida0e0b0xyg = t_confinement_pa1e1b1nwzida0e0b0xyg * fun.f_expand(bool_confinement_n, n_pos)

    ##7)Ensure that no feed supplies are outside the possible range
    nv_min_p6a1e1b1j1wzida0e0b0xyg = fun.f_expand(pinp.sheep['i_nv_lower_p6z'], left_pos=z_pos, left_pos2=p_pos, right_pos2=z_pos)
    nv_min_pa1e1b1j1wzida0e0b0xyg = np.take_along_axis(nv_min_p6a1e1b1j1wzida0e0b0xyg,a_p6_pa1e1b1nwzida0e0b0xyg,0)
    nv_max_p6a1e1b1j1wzida0e0b0xyg = fun.f_expand(pinp.sheep['i_nv_upper_p6z'], left_pos=z_pos, left_pos2=p_pos, right_pos2=z_pos)
    nv_max_pa1e1b1j1wzida0e0b0xyg = np.take_along_axis(nv_max_p6a1e1b1j1wzida0e0b0xyg,a_p6_pa1e1b1nwzida0e0b0xyg,0)
    feedsupply_std_stpa1e1b1nwzida0e0b0xyg = np.clip(feedsupply_std_stpa1e1b1nwzida0e0b0xyg, nv_min_pa1e1b1j1wzida0e0b0xyg, nv_max_pa1e1b1j1wzida0e0b0xyg)

    return feedsupply_std_stpa1e1b1nwzida0e0b0xyg, confinement_std_pa1e1b1nwzida0e0b0xyg, bool_confinement_n


def f1_pkl_feedsupply(lp_vars,r_vals,pkl_fs_info):
    '''
    Calculates the optimum feedsupply based on the stock activities selected and stores it in a pkl file.

    If an animal is not selected it gets the inputted fs for the current trial (this is not necessarily the xl fs). This
    method ensures less randomness because if an animal is not selected it will get the same fs as the previous trial.

    Note: Sires currently only have one w slice. There is no ability to optimise their lw. Thus optimal fs equals input fs.

    Use float32 to speed process when using lots of w.
    '''
    from . import ReportFunctions as rfun

    if sinp.structuralsa['i_fs_create_pkl']:
        ##inputs
        d_pos = sinp.stock['i_d_pos']
        n_pos = sinp.stock['i_n_pos']
        w_pos = sinp.stock['i_w_pos']
        p_pos = sinp.stock['i_p_pos']
        z_pos = sinp.stock['i_z_pos']

        ##access stock variables from lp output
        rfun.f_var_reshape(lp_vars,r_vals) #this func defines a global variable called d_vars
        stock_vars = rfun.d_vars['base']
        sire_numbers_qszg0 = stock_vars['sire_numbers_qszg0'].astype('float32')
        dams_numbers_qsk2tvanwziy1g1 = stock_vars['dams_numbers_qsk2tvanwziy1g1'].astype('float32')
        offs_numbers_qsk3k5tvnwziaxyg3 = stock_vars['offs_numbers_qsk3k5tvnwziaxyg3'].astype('float32')
        ###add singleton axis to line up with generator
        dams_numbers_qsk2tva1e1b1nwzida0e0b0xyg1 = dams_numbers_qsk2tvanwziy1g1[...,na,na,:,:,:,:,na,na,na,na,na,:,:]
        offs_numbers_qsk3k5tva1e1b1nwzida0e0b0xyg3 = offs_numbers_qsk3k5tvnwziaxyg3[...,na,na,na,:,:,:,:,na,:,na,na,:,:,:]
        ###weight by q,s&z. then remove the q&s axis (they can be removed here since the feedsupply doesn't have those axis)
        z_prob_qsk2tva1e1b1nwzida0e0b0xyg = fun.f_expand(r_vals['zgen']['z_prob_qsz'], z_pos, left_pos2=p_pos-3, right_pos2=z_pos)
        z_prob_qsk3k5tva1e1b1nwzida0e0b0xyg = fun.f_expand(r_vals['zgen']['z_prob_qsz'], z_pos, left_pos2=p_pos-4, right_pos2=z_pos)
        dams_numbers_k2tva1e1b1nwzida0e0b0xyg1 = fun.f_weighted_average(dams_numbers_qsk2tva1e1b1nwzida0e0b0xyg1, z_prob_qsk2tva1e1b1nwzida0e0b0xyg, axis=(0,1))
        offs_numbers_k3k5tva1e1b1nwzida0e0b0xyg3 = fun.f_weighted_average(offs_numbers_qsk3k5tva1e1b1nwzida0e0b0xyg3, z_prob_qsk3k5tva1e1b1nwzida0e0b0xyg, axis=(0,1))

        ##uncluster k axes on lp vars and convert v to p
        a_v_pa1e1b1nwzida0e0b0xyg1 = pkl_fs_info['a_v_pa1e1b1nwzida0e0b0xyg1']
        a_k2cluster_va1e1b1nwzida0e0b0xyg1 = pkl_fs_info['a_k2cluster_va1e1b1nwzida0e0b0xyg1']
        a_v_pa1e1b1nwzida0e0b0xyg3 = pkl_fs_info['a_v_pa1e1b1nwzida0e0b0xyg3']
        a_k3cluster_k3k5tva1e1b1nwzida0e0b0xyg3 = fun.f_expand(pkl_fs_info['a_k3cluster_da0e0b0xyg3'],p_pos-4,right_pos=d_pos)
        a_k5cluster_k5tva1e1b1nwzida0e0b0xyg3 = fun.f_expand(pkl_fs_info['a_k5cluster_da0e0b0xyg3'],p_pos-3,right_pos=d_pos)

        ##uncluster via association
        ###dams
        dams_numbers_tva1e1b1nwzida0e0b0xyg1 = np.take_along_axis(dams_numbers_k2tva1e1b1nwzida0e0b0xyg1, a_k2cluster_va1e1b1nwzida0e0b0xyg1[na,na], axis=0)[0] #slice of singleton k axis (now replaced by e&b)
        dams_numbers_tpa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(dams_numbers_tva1e1b1nwzida0e0b0xyg1, a_v_pa1e1b1nwzida0e0b0xyg1[na], axis=1)
        ###offs
        offs_numbers_k5tva1e1b1nwzida0e0b0xyg3 = np.take_along_axis(offs_numbers_k3k5tva1e1b1nwzida0e0b0xyg3, a_k3cluster_k3k5tva1e1b1nwzida0e0b0xyg3, axis=0)[0] #slice of singleton k3 axis (now replaced by d)
        offs_numbers_tva1e1b1nwzida0e0b0xyg3 = np.take_along_axis(offs_numbers_k5tva1e1b1nwzida0e0b0xyg3, a_k5cluster_k5tva1e1b1nwzida0e0b0xyg3, axis=0)[0] #slice of singleton k5 axis (now replaced by e&b)
        offs_numbers_tpa1e1b1nwzida0e0b0xyg3 = np.take_along_axis(offs_numbers_tva1e1b1nwzida0e0b0xyg3, a_v_pa1e1b1nwzida0e0b0xyg3[na], axis=1)

        ##access generator arrays
        ###feedsupply and confinement for current trial with w axis
        feedsupply_tpa1e1b1nwzida0e0b0xyg0 = pkl_fs_info['feedsupply_tpa1e1b1nwzida0e0b0xyg0'].astype('float32')
        feedsupply_tpa1e1b1nwzida0e0b0xyg1 = pkl_fs_info['feedsupply_tpa1e1b1nwzida0e0b0xyg1'].astype('float32')
        feedsupply_tpa1e1b1nwzida0e0b0xyg3 = pkl_fs_info['feedsupply_tpa1e1b1nwzida0e0b0xyg3'].astype('float32')
        confinementw_tpa1e1b1nwzida0e0b0xyg0 = pkl_fs_info['confinementw_tpa1e1b1nwzida0e0b0xyg0'].astype('float32')
        confinementw_tpa1e1b1nwzida0e0b0xyg1 = pkl_fs_info['confinementw_tpa1e1b1nwzida0e0b0xyg1'].astype('float32')
        confinementw_tpa1e1b1nwzida0e0b0xyg3 = pkl_fs_info['confinementw_tpa1e1b1nwzida0e0b0xyg3'].astype('float32')

        xl_feedsupply_pa1e1b1j2wzida0e0b0xyg0 = pkl_fs_info['xl_feedsupply_pa1e1b1j2wzida0e0b0xyg0'].astype('float32')
        xl_feedsupply_pa1e1b1j2wzida0e0b0xyg1 = pkl_fs_info['xl_feedsupply_pa1e1b1j2wzida0e0b0xyg1'].astype('float32')
        xl_feedsupply_pa1e1b1j2wzida0e0b0xyg3 = pkl_fs_info['xl_feedsupply_pa1e1b1j2wzida0e0b0xyg3'].astype('float32')
        xl_confinement_pa1e1b1nwzida0e0b0xyg0 = pkl_fs_info['xl_confinement_pa1e1b1nwzida0e0b0xyg0'].astype('float32')
        xl_confinement_pa1e1b1nwzida0e0b0xyg1 = pkl_fs_info['xl_confinement_pa1e1b1nwzida0e0b0xyg1'].astype('float32')
        xl_confinement_pa1e1b1nwzida0e0b0xyg3 = pkl_fs_info['xl_confinement_pa1e1b1nwzida0e0b0xyg3'].astype('float32')

        ##add w start (s) axis
        ###dams
        len_start_w1 = sinp.structuralsa['i_w_start_len1']
        dams_numbers_stpa1e1b1nwzida0e0b0xyg1 = np.moveaxis(fun.f_split_axis(dams_numbers_tpa1e1b1nwzida0e0b0xyg1, len_start_w1, w_pos), w_pos-1,0)
        feedsupply_stpa1e1b1nwzida0e0b0xyg1 = np.moveaxis(fun.f_split_axis(feedsupply_tpa1e1b1nwzida0e0b0xyg1, len_start_w1, w_pos), w_pos-1,0)
        confinementw_stpa1e1b1nwzida0e0b0xyg1 = np.moveaxis(fun.f_split_axis(confinementw_tpa1e1b1nwzida0e0b0xyg1, len_start_w1, w_pos), w_pos-1,0)
        ###offs
        len_start_w3 = sinp.structuralsa['i_w_start_len3']
        offs_numbers_stpa1e1b1nwzida0e0b0xyg3 = np.moveaxis(fun.f_split_axis(offs_numbers_tpa1e1b1nwzida0e0b0xyg3, len_start_w3, w_pos), w_pos-1,0)
        feedsupply_stpa1e1b1nwzida0e0b0xyg3 = np.moveaxis(fun.f_split_axis(feedsupply_tpa1e1b1nwzida0e0b0xyg3, len_start_w3, w_pos), w_pos-1,0)
        confinementw_stpa1e1b1nwzida0e0b0xyg3 = np.moveaxis(fun.f_split_axis(confinementw_tpa1e1b1nwzida0e0b0xyg3, len_start_w3, w_pos), w_pos-1,0)

        ##calculate the optimum feedsupply. Take weighted average across axis that are unwanted.
        ##note sires only ever have one w slice. There is no ability to optimise their lw. Thus optimal fs equals input fs.
        optimal_fs_tpa1e1b1j2wzida0e0b0xyg0 = xl_feedsupply_pa1e1b1j2wzida0e0b0xyg0[na] #add singleton t so that generator works
        optimal_fs_stpa1e1b1nwzida0e0b0xyg1 = fun.f_weighted_average(feedsupply_stpa1e1b1nwzida0e0b0xyg1,dams_numbers_stpa1e1b1nwzida0e0b0xyg1,w_pos,keepdims=True)
        optimal_fs_stpa1e1b1nwzida0e0b0xyg3 = fun.f_weighted_average(feedsupply_stpa1e1b1nwzida0e0b0xyg3,offs_numbers_stpa1e1b1nwzida0e0b0xyg3,w_pos,keepdims=True)

        ##update slices that have no numbers.
        ## In cases where pyomo doesn't select animals in the dvps the generator still needs a feedsupply.
        ## Update using the inputted feedsupply for the current trial (before taking weighted average) (this is not nessecarily the fs from xl).
        ## This method was chosen because it reduced randomness since if no animals are chosen the next trial gets the same fs as the current trial.
        ## This method wont work so well if the starting fs is not so good. To fix this ensure xl has decent starting fs and/or when generating fs force a small number of animals in all axes.
        ###dams
        optimal_fs_stpa1e1b1nwzida0e0b0xyg1 = fun.f_update(optimal_fs_stpa1e1b1nwzida0e0b0xyg1,fun.f_dynamic_slice(feedsupply_stpa1e1b1nwzida0e0b0xyg1,w_pos,0,1),
                                                           np.sum(dams_numbers_stpa1e1b1nwzida0e0b0xyg1,w_pos,keepdims=True)==0)
        ###offs
        optimal_fs_stpa1e1b1nwzida0e0b0xyg3 = fun.f_update(optimal_fs_stpa1e1b1nwzida0e0b0xyg3, fun.f_dynamic_slice(feedsupply_stpa1e1b1nwzida0e0b0xyg3,w_pos,0,1),
                                                           np.sum(offs_numbers_stpa1e1b1nwzida0e0b0xyg3,w_pos,keepdims=True)==0)

        ##populate the min and max slice of j2 axis - min and max slices of j2 are populated based on the same scale as the feedsupply inputs from excel
        ###dams
        j2_scale_pa1e1b1j2wzida0e0b0xyg1 = fun.f_divide(xl_feedsupply_pa1e1b1j2wzida0e0b0xyg1,
                                                        fun.f_dynamic_slice(xl_feedsupply_pa1e1b1j2wzida0e0b0xyg1, n_pos, 0, 1))
        optimal_fs_stpa1e1b1j2wzida0e0b0xyg1 = optimal_fs_stpa1e1b1nwzida0e0b0xyg1 * j2_scale_pa1e1b1j2wzida0e0b0xyg1
        ###offs
        j2_scale_pa1e1b1j2wzida0e0b0xyg3 = fun.f_divide(xl_feedsupply_pa1e1b1j2wzida0e0b0xyg3,
                                                        fun.f_dynamic_slice(xl_feedsupply_pa1e1b1j2wzida0e0b0xyg3, n_pos, 0, 1))
        optimal_fs_stpa1e1b1j2wzida0e0b0xyg3 = optimal_fs_stpa1e1b1nwzida0e0b0xyg3 * j2_scale_pa1e1b1j2wzida0e0b0xyg3
        
        ##calculate optimum confinement period - the proportion of animals in confinement must be greater than a cut off propn.
        cutoff = 0.75
        ###sires - no w optimisation therefore optimal = input
        optimal_confinement_tpa1e1b1nwzida0e0b0xyg0 = confinementw_tpa1e1b1nwzida0e0b0xyg0
        ###dams
        optimal_confinement_stpa1e1b1nwzida0e0b0xyg1 = fun.f_weighted_average(confinementw_stpa1e1b1nwzida0e0b0xyg1,dams_numbers_stpa1e1b1nwzida0e0b0xyg1,w_pos,keepdims=True)
        optimal_confinement_stpa1e1b1nwzida0e0b0xyg1 = optimal_confinement_stpa1e1b1nwzida0e0b0xyg1 > cutoff
        ###offs
        optimal_confinement_stpa1e1b1nwzida0e0b0xyg3 = fun.f_weighted_average(confinementw_stpa1e1b1nwzida0e0b0xyg3,offs_numbers_stpa1e1b1nwzida0e0b0xyg3,w_pos,keepdims=True)
        optimal_confinement_stpa1e1b1nwzida0e0b0xyg3 = optimal_confinement_stpa1e1b1nwzida0e0b0xyg3 > cutoff

        ##update confinement - For slices where no animals were selected in pyomo. They still need to be generated with a fs.
        ## update with the xl inputs - need to use xl inputs as default because otherwise if the model doesnt select confinement in one itteration it will not be able to select in a future itteration (unless one of the nut spread options is confinement)
        optimal_confinement_stpa1e1b1nwzida0e0b0xyg1 = fun.f_update(optimal_confinement_stpa1e1b1nwzida0e0b0xyg1, xl_confinement_pa1e1b1nwzida0e0b0xyg1,
                                                          np.sum(dams_numbers_tpa1e1b1nwzida0e0b0xyg1,w_pos,keepdims=True)==0)
        optimal_confinement_stpa1e1b1nwzida0e0b0xyg3 = fun.f_update(optimal_confinement_stpa1e1b1nwzida0e0b0xyg3, xl_confinement_pa1e1b1nwzida0e0b0xyg3,
                                                          np.sum(offs_numbers_tpa1e1b1nwzida0e0b0xyg3,w_pos,keepdims=True)==0)

        ##LTW adjustment
        sfw_ltwadj_pa1e1b1nwzida0e0b0xyg1 = pkl_fs_info['sfw_ltwadj_pa1e1b1nwzida0e0b0xyg1'].astype('float32')
        sfd_ltwadj_pa1e1b1nwzida0e0b0xyg1 = pkl_fs_info['sfd_ltwadj_pa1e1b1nwzida0e0b0xyg1'].astype('float32')
        sfw_ltwadj_pa1e1b1nwzida0e0b0xyg3 = pkl_fs_info['sfw_ltwadj_pa1e1b1nwzida0e0b0xyg3'].astype('float32')
        sfd_ltwadj_pa1e1b1nwzida0e0b0xyg3 = pkl_fs_info['sfd_ltwadj_pa1e1b1nwzida0e0b0xyg3'].astype('float32')

        ##pkl
        ##stick fs info into dict
        pkl_fs = dict()
        pkl_fs['fs'] = {}
        pkl_fs['confinement'] = {}
        pkl_fs['ltw_adj'] = {}
        pkl_fs['fs']['sire'] = optimal_fs_tpa1e1b1j2wzida0e0b0xyg0
        pkl_fs['fs']['dams'] = optimal_fs_stpa1e1b1j2wzida0e0b0xyg1
        pkl_fs['fs']['offs'] = optimal_fs_stpa1e1b1j2wzida0e0b0xyg3
        pkl_fs['confinement']['sire'] = optimal_confinement_tpa1e1b1nwzida0e0b0xyg0
        pkl_fs['confinement']['dams'] = optimal_confinement_stpa1e1b1nwzida0e0b0xyg1
        pkl_fs['confinement']['offs'] = optimal_confinement_stpa1e1b1nwzida0e0b0xyg3
        pkl_fs['ltw_adj']['sfw_ltwadj_pa1e1b1nwzida0e0b0xyg1'] = sfw_ltwadj_pa1e1b1nwzida0e0b0xyg1
        pkl_fs['ltw_adj']['sfd_ltwadj_pa1e1b1nwzida0e0b0xyg1'] = sfd_ltwadj_pa1e1b1nwzida0e0b0xyg1
        pkl_fs['ltw_adj']['sfw_ltwadj_pa1e1b1nwzida0e0b0xyg3'] = sfw_ltwadj_pa1e1b1nwzida0e0b0xyg3
        pkl_fs['ltw_adj']['sfd_ltwadj_pa1e1b1nwzida0e0b0xyg3'] = sfd_ltwadj_pa1e1b1nwzida0e0b0xyg3
        pkl_fs['pkl_condensed_values'] = pkl_fs_info['pkl_condensed_values']

        return pkl_fs
