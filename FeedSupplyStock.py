'''
Stock feedsupply construction

author: young

Feedsupply is represented as a nutritive value (NV). There is also an input which controls if the animal
is in confinement or not.

Feedsupply can be generated in two ways

    #. from inputs in Property.xl and Structural.xl
    #. from pkl file that is generated from running the model with multiple w options.

Feedsupply inputs contain a base feedsupply (i_feedoptions_r1pj0) for different animals. The base feed supply
used for each animal in the generator is selected using ia_r1_zig?.

For dams (has not been hooked up for offs) there is possible further adjustment (i_feedsupply_adj_options_r2p)
for LSLN (litter size lactation number - b1 axis) and weaning
age (a1 axis). Typically this is not used. Instead the user runs AFO with multiple w and allows the model to
generate the optimal feedsupply which is used for subsequent runs.

Generating the feedsupply from the optimal nutrition pattern selected by the model means that the feedsupply
can easily be optimised for different axis that are not included in the inputs (eg t axis and z axis).
This is carried out in the feedsupply creation experiment (in exp.xl). Depending on the number of w that
can be included in the model it may take several trials to converge on the optimal feedsupply.
Optimising the duration of confinement feeding is imprecise because the optimisation occurs at the DVP level
and this timestep is likely too long. Therefore, the user needs to test including confinement in a range of feed
periods for different stock classes. This is controlled using the input i_confinement_r1p6z.

When using the pkl file as the source of the feedsupply inputs it is possible to control whether the inclusion
of confinement is controlled by the previous optimum solution (from the pkl file) or from the
input i_confinement_r1p6z. This allows the inclusion of confinement feeding in subsequent trials even if it wasn't
selected in the previous optimum solution.
This also allows confinement to be included in the N1 model without forcing it to occur all year.

Confinement feeding is good to represent separately for a few reason:

    #. Feeding a given quantity of energy per day in confinement results in better production because less energy
       is expended grazing.
    #. Confinement feeding is represented in a separate feed pool because a restricted intake of supplement that meets
       the energy requirements of an animal results in slack volume. If in the same feed pool this slack volume could
       be used to consume greater quantities of poor quality feed such as dry pasture or stubble.
    #. The confinement pool is not adjusted for effective MEI, which is to allow for the lower efficiency of feeding
       if feed quality is greater than the quality required to meet the target for the given pool.
    #. Confinement feeding should incur a capital cost of the feed lot and the labour requirement should reflect
       feedlot conditions rather than paddock feeding conditions.

Likely process to calibrate feedsupply: feedsupply can be calibrated prior to an analysis using multiple n
slices and once the optimum is identified the model can be set back to N1. This might take multiple iterations
using the big model. The feedsupply will need to be generated for each group of trials that have different
active sheep axis (ie if the analysis is comparing scanning options then there would need to be a feedsupply
for scan=0 and scan=1 because the clustering is different). A further challenge is to optimise when to confinement
feed. Optimising the NV in confinement is similar to optimising in the paddock and requires multiple confinement n.





'''


import numpy as np

import PropertyInputs as pinp
import UniversalInputs as uinp
import StructuralInputs as sinp
import StockFunctions as sfun
import Functions as fun
import FeedsupplyFunctions as fsfun
import SeasonalFunctions as zfun




#todo supp feeding in confinement incurs the same costs as paddock feeding. this should be changed. it should also incur some capital cost.

def f1_stock_fs(cr_sire,cr_dams,cr_offs,cu0_sire,cu0_dams,cu0_offs,a_p6_pa1e1b1nwzida0e0b0xyg,
                 period_between_weanprejoin_pa1e1b1nwzida0e0b0xyg1,
                 scan_management_pa1e1b1nwzida0e0b0xyg1, gbal_management_pa1e1b1nwzida0e0b0xyg1, wean_management_pa1e1b1nwzida0e0b0xyg1,
                 a_n_pa1e1b1nwzida0e0b0xyg1, a_n_pa1e1b1nwzida0e0b0xyg3, mask_p_offs_p, len_p):

    #########
    #inputs #
    #########
    na=np.newaxis

    ##pos
    a0_pos = sinp.stock['i_a0_pos']
    a1_pos = sinp.stock['i_a1_pos']
    b0_pos = sinp.stock['i_b0_pos']
    b1_pos = sinp.stock['i_b1_pos']
    d_pos = sinp.stock['i_d_pos']
    e1_pos = sinp.stock['i_e1_pos']
    i_pos = sinp.stock['i_i_pos']
    n_pos = sinp.stock['i_n_pos']
    p_pos = sinp.stock['i_p_pos']
    x_pos = sinp.stock['i_x_pos']
    z_pos = sinp.stock['i_z_pos']

    ##len
    len_a1 = np.count_nonzero(pinp.sheep['i_mask_a'])


    ##nut
    n_fs_sire = sinp.structuralsa['i_n0_len']
    n_fs_dams = sinp.structuralsa['i_n1_len']
    n_fs_offs = sinp.structuralsa['i_n3_len']

    ##pasture params
    cu3 = uinp.pastparameters['i_cu3_c4'][...,pinp.sheep['i_pasture_type']].astype(float)#have to convert from object to float so it doesnt chuck error in np.exp (np.exp can't handle object arrays)
    cu4 = uinp.pastparameters['i_cu4_c4'][...,pinp.sheep['i_pasture_type']].astype(float)#have to convert from object to float so it doesnt chuck error in np.exp (np.exp can't handle object arrays)

    ##legume proportion in each period
    legume_p6a1e1b1nwzida0e0b0xyg = fun.f_expand(pinp.sheep['i_legume_p6z'], z_pos, move=True, source=0, dest=-1,
                                                 left_pos2=p_pos, right_pos2=z_pos)
    ##estimated foo and dmd for the feed periods (p6) periods
    paststd_foo_p6a1e1b1j0wzida0e0b0xyg = fun.f_expand(pinp.sheep['i_paststd_foo_zp6j0'],z_pos,move=True,source=0,
                                                       dest=2, left_pos2=n_pos,right_pos2=z_pos,left_pos3=p_pos,
                                                       right_pos3=n_pos)
    pasture_stage_p6a1e1b1j0wzida0e0b0xyg = fun.f_expand(pinp.sheep['i_pasture_stage_p6z'], z_pos, move=True, source=0,
                                                         dest=-1, left_pos2=p_pos, right_pos2=z_pos)  # z is treated in next step
    ##foo corrected to hand shears and estimated height - the z axis is also treated in this step
    paststd_foo_p6a1e1b1j0wzida0e0b0xyg0, paststd_hf_p6a1e1b1j0wzida0e0b0xyg0 = fsfun.f_foo_convert(cu3, cu4,
                                                                                     paststd_foo_p6a1e1b1j0wzida0e0b0xyg,
                                                                                     pasture_stage_p6a1e1b1j0wzida0e0b0xyg,
                                                                                     legume_p6a1e1b1nwzida0e0b0xyg, cr_sire,
                                                                                     z_pos=sinp.stock['i_z_pos'], treat_z=True)
    paststd_foo_p6a1e1b1j0wzida0e0b0xyg1, paststd_hf_p6a1e1b1j0wzida0e0b0xyg1 = fsfun.f_foo_convert(cu3, cu4,
                                                                                     paststd_foo_p6a1e1b1j0wzida0e0b0xyg,
                                                                                     pasture_stage_p6a1e1b1j0wzida0e0b0xyg,
                                                                                     legume_p6a1e1b1nwzida0e0b0xyg, cr_dams,
                                                                                     z_pos=sinp.stock['i_z_pos'], treat_z=True)
    paststd_foo_p6a1e1b1j0wzida0e0b0xyg3, paststd_hf_p6a1e1b1j0wzida0e0b0xyg3 = fsfun.f_foo_convert(cu3, cu4,
                                                                                     paststd_foo_p6a1e1b1j0wzida0e0b0xyg,
                                                                                     pasture_stage_p6a1e1b1j0wzida0e0b0xyg,
                                                                                     legume_p6a1e1b1nwzida0e0b0xyg, cr_offs,
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
    about the feed consumed eg FOO and DMD because these factors impact things like energy required (eg less feed means
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
    a_r_zida0e0b0xyg0 = sfun.f1_g2g(pinp.sheep['ia_r1_zig0'], 'sire',i_pos, swap=True, condition=pinp.sheep['i_masksire_i']
                                    , axis=i_pos).astype(int)
    a_r_zida0e0b0xyg1 = sfun.f1_g2g(pinp.sheep['ia_r1_zig1'], 'dams',i_pos, swap=True, condition=pinp.sheep['i_mask_i']
                                    , axis=i_pos).astype(int)
    a_r_zida0e0b0xyg3 = sfun.f1_g2g(pinp.sheep['ia_r1_zig3'], 'offs',i_pos, swap=True, condition=pinp.sheep['i_mask_i']
                                    , axis=i_pos).astype(int)
    ###feed adjustment for dams
    a_r2_k0e1b1nwzida0e0b0xyg1 = sfun.f1_g2g(pinp.sheep['ia_r2_k0ig1'], 'dams',i_pos, swap=True, left_pos2=a1_pos
                                             , right_pos2=i_pos, condition=pinp.sheep['i_mask_i'], axis=i_pos)
    a_r2_k1b1nwzida0e0b0xyg1 = sfun.f1_g2g(pinp.sheep['ia_r2_k1ig1'], 'dams', i_pos, swap=True, left_pos2=e1_pos
                                           , right_pos2=i_pos, condition=pinp.sheep['i_mask_i'], axis=i_pos)
    a_r2_spk0k1k2nwzida0e0b0xyg1 = sfun.f1_g2g(pinp.sheep['ia_r2_sk2ig1'], 'dams',i_pos, left_pos2=b1_pos, right_pos2=i_pos
                                               , left_pos3=p_pos-1, right_pos3=b1_pos, condition=pinp.sheep['i_mask_i']
                                               , axis=i_pos, move=True, source=0, dest=2)  #add axis between g and i and i and b1
    ###feed adjustment for offs
    a_r2_idk0e0b0xyg3 = sfun.f1_g2g(pinp.sheep['ia_r2_ik0g3'], 'offs',a0_pos, left_pos2=i_pos, right_pos2=a0_pos
                                    , condition=pinp.sheep['i_mask_i'], axis=i_pos)
    a_r2_ik3a0e0b0xyg3 = sfun.f1_g2g(pinp.sheep['ia_r2_ik3g3'], 'offs',d_pos,  condition=pinp.sheep['i_mask_i'], axis=i_pos)
    a_r2_ida0e0k4xyg3 = sfun.f1_g2g(pinp.sheep['ia_r2_ik4g3'], 'offs',b0_pos, left_pos2=i_pos, right_pos2=b0_pos
                                    , condition=pinp.sheep['i_mask_i'], axis=i_pos)  #add axis between g and b0 and b0 and i
    a_r2_ida0e0b0k5yg3 = sfun.f1_g2g(pinp.sheep['ia_r2_ik5g3'], 'offs',x_pos, left_pos2=i_pos, right_pos2=x_pos
                                     , condition=pinp.sheep['i_mask_i'], axis=i_pos)  #add axis between g and b0 and b0 and i

    ##std feed options
    feedsupply_options_r1j2p = pinp.feedsupply['i_feedsupply_options_r1j2p'][...,0:len_p].astype(np.float) #slice off extra p periods so it is the same length as the sim periods
    ##confinement - True/False on confinement feeding. This controls which generator periods confinement feeding occurs.
    ## this input is required so that confinement can be included in n1 model, without forcing the animal into confinement for the whole year.
    ## This means you can have a given level of NV and it can be either in the paddock or in confinement.
    ## This input works inconjunction with i_confinement_n (see in the feed supply section)
    confinement_options_r1p6z = pinp.feedsupply['i_confinement_options_r1p6z'].astype(np.float) #slice off extra p periods so it is the same length as the sim periods
    ##feed supply adjustment
    feedsupply_adj_options_r2p = pinp.feedsupply['i_feedsupply_adj_options_r2p'][:,0:len_p].astype(np.float) #slice off extra p periods so it is the same length as the sim periods
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
    #### eg the optimum feedsupply prior to scanning may change depending on whether singles and twins are identified
    #todo the above comment is correct however, it is not represented in the inputs of the model. Undiff/mated is always 0 regardless of the scanning level
    #####take along the scan axis then remove the singleton scan axis with [0]
    a_r2_pk0k1k2nwzida0e0b0xyg1 = np.take_along_axis(a_r2_spk0k1k2nwzida0e0b0xyg1, scan_management_pa1e1b1nwzida0e0b0xyg1[na,...], axis=0)[0]
    ####select feedsupply adjustment option for each b slice based on the 'k2 input cluster' association.
    a_r2_lsln_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(a_r2_pk0k1k2nwzida0e0b0xyg1, a_k2_pa1e1b1nwzida0e0b0xyg1, b1_pos)

    ###d) todo come back to offs (remember gender needs to be masked)
    # t_fs_agedam_pj2zida0e0b0xg3 = t_fs_agedam_pj2zik3k0k4k5g3
    # t_fs_ageweaned_pj2zida0e0b0xg3 = t_fs_ageweaned_pj2zik3k0k4k5g3
    # t_fs_btrt_pj2zida0e0b0xg3 = t_fs_btrt_pj2zik3k0k4k5g3
    # t_fs_gender_pj2zida0e0b0xg3 = t_fs_gender_pj2zik3k0k4k5g3


    ##3) calculate the feedsupply adjustment for each sheep class
    feedsupply_adj_options_r2pa1e1b1nwzida0e0b0xyg1 = fun.f_expand(feedsupply_adj_options_r2p,p_pos) #add other axis as singleton
    ###a)wean (take along the r2 axis and then remove the singleton axis with [0])
    t_fs_ageweaned_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(feedsupply_adj_options_r2pa1e1b1nwzida0e0b0xyg1
                                                                , a_r2_wean_pa1e1b1nwzida0e0b0xyg1[na,...], axis=0)[0]
    ###b)oestrus (take along the r2 axis and then remove the singleton axis with [0])
    # t_fs_cycle_pa1e1b1j2wzida0e0b0xyg1 = np.take_along_axis(feedsupply_adj_options_r2pa1e1b1nwzida0e0b0xyg1
    #                                                         , a_r2_oestrus_pa1e1b1nwzida0e0b0xyg1[na,...], axis=0)[0]
    ###c)lsln (take along the r2 axis and then remove the singleton axis with [0])
    t_fs_lsln_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(feedsupply_adj_options_r2pa1e1b1nwzida0e0b0xyg1
                                                           , a_r2_lsln_pa1e1b1nwzida0e0b0xyg1[na,...], axis=0)[0]

    # t_fs_agedam_pa1e1b1j2wzik3a0e0b0xyg3 =
    # t_fs_ageweaned_pa1e1b1j2wzidk0e0b0xyg3 =
    # t_fs_btrt_a1e1b1j2wzida0e0k4xyg3 =
    # t_fs_gender_pa1e1b1j2wzida0e0b0k5yg3 =



    ##4) add adjustment to std pattern (the adjustment is broadcast across j2 (the standard, minimum and maximum))
    ##feedsupply is clipped below to ensure it is within a feasible range.
    ##Note: the adjustment is in FS units (not in MJ/kg)
    t_feedsupply_pa1e1b1j2wzida0e0b0xyg1 = (t_feedsupply_pa1e1b1j2wzida0e0b0xyg1 + t_fs_ageweaned_pa1e1b1nwzida0e0b0xyg1
                                                   + t_fs_lsln_pa1e1b1nwzida0e0b0xyg1) #can't use += for some reason
    # t_feedsupply_pa1e1b1j2wzida0e0b0xyg3 = (t_feedsupply_pa1e1b1j2wzida0e0b0xyg3 + t_fs_agedam_pj2zida0e0b0xg3
    #                                             + t_fs_ageweaned_pj2zida0e0b0xg3 + t_fs_gender_pj2zida0e0b0xg3)


    ##5)Convert the ‘j2’ axis to an ‘n’ axis using the nut_spread inputs.
    ## activate n axis for confinement control (controls if a nutrition pattern is in confinement - note
    ## if i_confinement_n? is set to True the generator periods confinement occurs is controlled by i_confinement_options_r1p6z.
    feedsupply_std_pa1e1b1nwzida0e0b0xyg0, confinement_std_pa1e1b1nwzida0e0b0xyg0, bool_confinement_g0_n =  \
        f1_j2_to_n(t_feedsupply_pa1e1b1j2wzida0e0b0xyg0, t_confinement_pa1e1b1nwzida0e0b0xyg0, nv_p6a1e1b1j1wzida0e0b0xyg0,
                   a_p6_pa1e1b1nwzida0e0b0xyg, sinp.structuralsa['i_nut_spread_n0'], sinp.structuralsa['i_confinement_n0'],
                   n_fs_sire)

    feedsupply_std_pa1e1b1nwzida0e0b0xyg1, confinement_std_pa1e1b1nwzida0e0b0xyg1, bool_confinement_g1_n = \
        f1_j2_to_n(t_feedsupply_pa1e1b1j2wzida0e0b0xyg1, t_confinement_pa1e1b1nwzida0e0b0xyg1, nv_p6a1e1b1j1wzida0e0b0xyg1,
                   a_p6_pa1e1b1nwzida0e0b0xyg, sinp.structuralsa['i_nut_spread_n1'], sinp.structuralsa['i_confinement_n1'],
                   n_fs_dams)
    feedsupply_std_pa1e1b1nwzida0e0b0xyg3, confinement_std_pa1e1b1nwzida0e0b0xyg3, bool_confinement_g3_n = \
        f1_j2_to_n(t_feedsupply_pa1e1b1j2wzida0e0b0xyg3, t_confinement_pa1e1b1nwzida0e0b0xyg3, nv_p6a1e1b1j1wzida0e0b0xyg3,
                   a_p6_pa1e1b1nwzida0e0b0xyg[mask_p_offs_p], sinp.structuralsa['i_nut_spread_n3'], sinp.structuralsa['i_confinement_n3'],
                   n_fs_offs)

    ##6) expand feedsupply for all w slices - see google doc for more info
    ###convert feedsupply from having an active n axis to active w axis by applying association
    #### each slice of w has a combination of nutrition levels during the nutrition cycle and that is specified in the association a_n_pw
    feedsupplyw_pa1e1b1nwzida0e0b0xyg0 = feedsupply_std_pa1e1b1nwzida0e0b0xyg0 #^only one n slice so doesnt need following code yet: np.take_along_axis(feedsupply_std_pa1e1b1nwzida0e0b0xyg0,a_n_pa1e1b1nwzida0e0b0xyg0,axis=n_pos)
    feedsupplyw_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(feedsupply_std_pa1e1b1nwzida0e0b0xyg1, a_n_pa1e1b1nwzida0e0b0xyg1, axis=n_pos)
    feedsupplyw_pa1e1b1nwzida0e0b0xyg3 = np.take_along_axis(feedsupply_std_pa1e1b1nwzida0e0b0xyg3, a_n_pa1e1b1nwzida0e0b0xyg3, axis=n_pos)

    ##convert n to w for confinement control
    confinementw_pa1e1b1nwzida0e0b0xyg0 = confinement_std_pa1e1b1nwzida0e0b0xyg0 #^only one n slice so doesnt need following code yet: np.take_along_axis(feedsupply_std_pa1e1b1nwzida0e0b0xyg0,a_n_pa1e1b1nwzida0e0b0xyg0,axis=n_pos)
    confinementw_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(confinement_std_pa1e1b1nwzida0e0b0xyg1, a_n_pa1e1b1nwzida0e0b0xyg1, axis=n_pos)
    confinementw_pa1e1b1nwzida0e0b0xyg3 = np.take_along_axis(confinement_std_pa1e1b1nwzida0e0b0xyg3, a_n_pa1e1b1nwzida0e0b0xyg3, axis=n_pos)


    return legume_p6a1e1b1nwzida0e0b0xyg, bool_confinement_g0_n, bool_confinement_g1_n, bool_confinement_g3_n, \
           nv_p6a1e1b1j1wzida0e0b0xyg0,foo_p6a1e1b1j1wzida0e0b0xyg0,dmd_p6a1e1b1j1wzida0e0b0xyg0,supp_p6a1e1b1j1wzida0e0b0xyg0,\
           nv_p6a1e1b1j1wzida0e0b0xyg1,foo_p6a1e1b1j1wzida0e0b0xyg1,dmd_p6a1e1b1j1wzida0e0b0xyg1,supp_p6a1e1b1j1wzida0e0b0xyg1,\
           nv_p6a1e1b1j1wzida0e0b0xyg3,foo_p6a1e1b1j1wzida0e0b0xyg3,dmd_p6a1e1b1j1wzida0e0b0xyg3,supp_p6a1e1b1j1wzida0e0b0xyg3,\
           feedsupplyw_pa1e1b1nwzida0e0b0xyg0, feedsupplyw_pa1e1b1nwzida0e0b0xyg1, feedsupplyw_pa1e1b1nwzida0e0b0xyg3, \
           confinementw_pa1e1b1nwzida0e0b0xyg0, confinementw_pa1e1b1nwzida0e0b0xyg1, confinementw_pa1e1b1nwzida0e0b0xyg3



def f1_j2_to_n(t_feedsupply_pa1e1b1j2wzida0e0b0xyg, t_confinement_pa1e1b1nwzida0e0b0xyg, nv_p6a1e1b1j1wzida0e0b0xyg,
               a_p6_pa1e1b1nwzida0e0b0xyg, i_nut_spread_n, i_confinement_n, n_fs):
    na = np.newaxis
    n_pos = sinp.stock['i_n_pos']
    ### the nut_spread inputs are the proportion of std and min or max feed supply.
    ### Unless nut_spread is greater than 3 in which case the value becomes the actual feed supply
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
    t_feedsupply_pa1e1b1j2nwzida0e0b0xyg = np.expand_dims(t_feedsupply_pa1e1b1j2wzida0e0b0xyg, axis = n_pos) #add n axis
    feedsupply_std_pa1e1b1nwzida0e0b0xyg = np.sum(t_feedsupply_pa1e1b1j2nwzida0e0b0xyg * nut_mult_pk0k1k2j2nwzida0e0b0xyg, axis = n_pos-1 ) #sum j axis, minus 1 because n axis was added therefore shifting j2 position (it was originally in the same place). Sum across j2 axis and leave just the n axis

    ###c activate n axis on confinement control
    confinement_std_pa1e1b1nwzida0e0b0xyg = t_confinement_pa1e1b1nwzida0e0b0xyg * fun.f_expand(bool_confinement_n, n_pos)
    # confinement_std_pa1e1b1nwzida0e0b0xyg1 = t_confinement_pa1e1b1nwzida0e0b0xyg1 * fun.f_expand(bool_confinement_g1_n, n_pos)
    # confinement_std_pa1e1b1nwzida0e0b0xyg3 = t_confinement_pa1e1b1nwzida0e0b0xyg3 * fun.f_expand(bool_confinement_g3_n, n_pos)

    ##7)Ensure that no feed supplies are outside the possible range - j1[0] is the lowest NV as determined by the poorest feed specified in the j0 inputs. j1[-1] is ad lib supplement so will equate to i_md_supp
    nv_min_p6a1e1b1j1wzida0e0b0xyg = fun.f_dynamic_slice(nv_p6a1e1b1j1wzida0e0b0xyg, axis=n_pos, start=0, stop=1)
    nv_min_pa1e1b1j1wzida0e0b0xyg = np.take_along_axis(nv_min_p6a1e1b1j1wzida0e0b0xyg,a_p6_pa1e1b1nwzida0e0b0xyg,0)
    nv_max_p6a1e1b1j1wzida0e0b0xyg = fun.f_dynamic_slice(nv_p6a1e1b1j1wzida0e0b0xyg, axis=n_pos, start=-1, stop=None)
    nv_max_pa1e1b1j1wzida0e0b0xyg = np.take_along_axis(nv_max_p6a1e1b1j1wzida0e0b0xyg,a_p6_pa1e1b1nwzida0e0b0xyg,0)
    feedsupply_std_pa1e1b1nwzida0e0b0xyg = np.clip(feedsupply_std_pa1e1b1nwzida0e0b0xyg, nv_min_pa1e1b1j1wzida0e0b0xyg, nv_max_pa1e1b1j1wzida0e0b0xyg)

    return feedsupply_std_pa1e1b1nwzida0e0b0xyg, confinement_std_pa1e1b1nwzida0e0b0xyg, bool_confinement_n


