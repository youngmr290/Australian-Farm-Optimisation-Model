'''
THis is where you can set up constraints on individual variables
withing a set of variables. This will act as bounds.

bounds are all controlled from this module
'''

import numpy as np
import pandas as pd
import time
import pyomo.environ as pe

from . import Functions as fun
from . import Sensitivity as sen
from . import UniversalInputs as uinp
from . import StructuralInputs as sinp
from . import PropertyInputs as pinp
from . import PhasePyomo as phspy
from . import MachPyomo as macpy
from . import LabourPyomo as labpy
from . import LabourPhasePyomo as lphspy
from . import PasturePyomo as paspy
from . import SupFeedPyomo as suppy
from . import CropResiduePyomo as stubpy
from . import StockPyomo as stkpy
from . import CropGrazingPyomo as cgzpy
from . import SaltbushPyomo as slppy


'''
Bounds

note:
-forcing sale/retention of drys is done in the stock module (there are inputs which user can control this with)
-wether sale age bound is in stock module (controlled via SAV)
'''



def f1_boundarypyomo_local(params, model):
    lmu_mask = pinp.lmu_mask
    mask_r = pinp.rot_mask_r
    z_mask = pinp.general['i_mask_z']

    ##set bounds to include
    bounds_inc = True #controls all bounds (typically on)
    rot_lobound_inc = np.any(sen.sav['rot_lobound_rl'] != '-')  #controls rot bound
    tree_area_inc = np.any(sen.sav['bnd_tree_area_l'][lmu_mask] != '-') #control the area of salt land pasture
    slp_area_inc = np.any(sen.sav['bnd_slp_area_l'][lmu_mask] != '-') #control the area of salt land pasture
    sb_upbound_inc = np.any(sen.sav['bnd_sb_consumption_p6'] != '-') #upper bound on the quantity of saltbush consumed
    sup_lobound_inc = False #controls sup feed bound
    sup_per_dse_bnd_inc = sen.sav['bnd_sup_per_dse'] != '-' #lower bound dams #controls sup per dse bound
    crop_grazing_intensity_bnd_inc = sen.sav['bnd_crop_grazing_intensity'] != '-'
    total_dams_eqbound_inc = sen.sav['bnd_total_dams'] != '-' #bound the total number of dams at prejoining
    dams_lobound_inc = fun.f_sa(False, sen.sav['bnd_lo_dam_inc'], 5) #lower bound dams
    dams_upbound_inc = (sen.sav['bnd_up_dams_K2tog1']!="-").any() or (sen.sav['bnd_up_dams_K2tVg1']!="-").any()  #upper bound on dams
    offs_lobound_inc = fun.f_sa(False, sen.sav['bnd_lo_off_inc'], 5) #lower bound offs
    offs_upbound_inc = fun.f_sa(False, sen.sav['bnd_up_off_inc'], 5) #upper bound on offs
    prog_upbound_inc = any(value != 999999 for value in params['stock']['p_prog_upbound'].values()) #upper bound on prog
    total_dams_scanned_bound_inc = np.any(sen.sav['bnd_total_dams_scanned'] != '-') #equal to bound on the total number of mated dams at scanning
    force_5yo_retention_inc = np.any(sen.sav['bnd_propn_dam5_retained'] != '-') #force a propn of 5yo dams to be retained.
    propn_mated_inc = np.any(sen.sav['bnd_propn_dams_mated_og1'] != '-')
    w_set_inc = fun.f_sa(False, sen.sav['propn_mated_w_inc'], 5)
    bnd_propn_dams_mated_inc = propn_mated_inc and not(w_set_inc) #include bnd_propn_mated without a w set.
    bnd_propn_dams_mated_w_inc = propn_mated_inc and w_set_inc #include bnd_propn_mated with a w set.
    bnd_sale_twice_drys_inc = fun.f_sa(False, sen.sav['bnd_sale_twice_dry_inc'], 5) #proportion of drys sold (can be sold at either sale opp)
    bnd_propn_singles_inc = np.any(sen.sav['min_propn_singles_sold_og1'] != '-') #include bound on the minimum proportion of singles sold
    bnd_propn_twins_inc = np.any(sen.sav['min_propn_twins_sold_og1'] != '-') #include bound on the minimum proportion of twins sold
    bnd_dry_retained_inc = fun.f_sa(False, np.any(pinp.sheep['i_dry_retained_forced_o']), 5) #force the retention of drys in t[0] (t[1] is handled in the generator.
    sr_bound_inc = np.any(sen.sav['bnd_sr_Qt'] != '-') #controls sr bound
    lw_bound_inc = sen.sav['bnd_lw_change'] != '-' #controls lw bound
    total_pasture_bound_inc = np.any(sen.sav['bnd_total_pas_area_percent_q'] != '-') #bound on total pasture (hence also total crop)
    pasture_bound_inc = np.any(sen.sav['bnd_pas_area_percent_t'] != '-') #bound area of each pasture type
    legume_area_bound_inc = sen.sav['bnd_total_legume_area_percent'] != '-'  #bound on total legume
    pasture_lmu_bound_inc = np.any(sen.sav['bnd_pas_area_l'] != '-')
    landuse_bound_inc = np.any(sen.sav['bnd_landuse_area_klz'] != '-') #bound on area of each landuse (which is the sum of all the phases for that landuse)
    cont_phase_bound_inc = np.any(sen.sav['max_yr_cont_k'] != '-') #bound to limit the number of years of a contunuous rotation
    crop_area_bound_inc = np.any(sen.sav['bnd_crop_area_qk1'] != '-') or np.any(sen.sav['bnd_crop_area_percent_qk1'] != '-')  # controls if crop area bnd is included.(which is the sum of all the phases for that crop)
    biomass_graze_bound_inc = np.any(sen.sav['bnd_biomass_graze_k1'] != '-')   # controls if biomass grazed bnd is included.(which is the proportion of crop biomass that is grazed)
    #todo need to make this input below in uinp. Then test the constraint works as expected.
    emissions_bnd_inc = False#uinp.emissions['co2e_limit']>0  # controls if total farm emissions are constrained.


    if bounds_inc:
        print('bounds implemented - make sure they are correct')

        '''Process:
            1. initialise arrays which are used as bounds
            2. set the bound here, can do this like assigning any value to numpy.
               These could be adjusted with SA values if you want to alter the bounds for different trials
               - The forced sale or retain of drys is controlled by livestock generator inputs
            3. ravel and zip bound and dict
            4.build the constraint '''

        ##params used in multiple bounds
        model.p_mask_dams = pe.Param(model.s_k2_birth_dams, model.s_sale_dams, model.s_dvp_dams, model.s_lw_dams, model.s_season_types
                                     , model.s_groups_dams, default=0, initialize=params['stock']['p_mask_dams'])
        model.p_mask_offs = pe.Param(model.s_k3_damage_offs, model.s_dvp_offs, model.s_lw_offs, model.s_season_types
                                     , model.s_gender, model.s_groups_dams, default=0, initialize=params['stock']['p_mask_offs'])

        ##rotations
        ###build bound if turned on
        if rot_lobound_inc:
            ###keys to build arrays
            arrays = [model.s_phases, model.s_lmus]
            index_rl = fun.cartesian_product_simple_transpose(arrays)
            ###build array
            #rot_lobound_rl = np.zeros((len(model.s_phases), len(model.s_lmus)))
            ###set the bound
            sen.sav['rot_lobound_rl'] = sen.sav['rot_lobound_rl'][0:len(mask_r),:]
            rot_lobound_rl = fun.f_sa(np.array([0],dtype=float), sen.sav['rot_lobound_rl'][mask_r,:][:,lmu_mask], 5)
            # rot_lobound_rl[4,0] = 70 #fodder lmu2
            # rot_lobound_rl[0,0] = 150 #AAAAAa
            # rot_lobound_rl[0,1] = 1230 #AAAAAa
            # rot_lobound_rl[0,2] = 750 #AAAAAa
            # rot_lobound_rl[2,1] = 570
            # rot_lobound_rl[2,2] = 20
            # rot_lobound_rl[9,1] = 11
            # rot_lobound_rl[9,2] = 87
            # rot_lobound_rl[15,1] = 11
            # rot_lobound_rl[15,2] = 87
            ###ravel and zip bound and dict
            rot_lobound = rot_lobound_rl.ravel()
            tup_rl = tuple(map(tuple, index_rl))
            rot_lobound = dict(zip(tup_rl, rot_lobound))
            ###constraint
            l_p7 = list(model.s_season_periods)
            def rot_lo_bound(model, q, s, p7, r, l, z):
                if p7 == l_p7[-1] and pe.value(model.p_wyear_inc_qs[q, s]):
                    return model.v_phase_area[q, s, p7, z, r, l] >= rot_lobound[r,l]
                else:
                    return pe.Constraint.Skip
            model.con_rotation_lobound = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_periods, model.s_phases, model.s_lmus, model.s_season_types, rule=rot_lo_bound,
                                                    doc='lo bound for the number of each phase')


        ##tree area
        if tree_area_inc:
            ###set the bound
            tree_area_bnd_l = fun.f_sa(np.array([999999],dtype=float), sen.sav['bnd_tree_area_l'][lmu_mask], 5) #999999 is arbitrary default value which mean skip constraint
            ###ravel and zip bound and dict
            tree_area = dict(zip(model.s_lmus, tree_area_bnd_l))
            ###constraint
            l_p7 = list(model.s_season_periods)
            def tree_area_bound(model, l):
                if tree_area[l] != 999999:
                    return model.v_tree_area_l[l] == tree_area[l]
                else:
                    return pe.Constraint.Skip
            model.con_tree_area_bound = pe.Constraint(model.s_lmus, rule=tree_area_bound,
                                                    doc='bound for the area of tree plantations on each lmu')


        ##salt land pasture area
        if slp_area_inc:
            ###set the bound
            slp_area_bnd_l = fun.f_sa(np.array([999999],dtype=float), sen.sav['bnd_slp_area_l'][lmu_mask], 5) #999999 is arbitrary default value which mean skip constraint
            ###ravel and zip bound and dict
            slp_area = dict(zip(model.s_lmus, slp_area_bnd_l))
            ###constraint
            l_p7 = list(model.s_season_periods)
            def slp_area_bound(model, q, s, z, l):
                if pe.value(model.p_wyear_inc_qs[q, s]) and slp_area[l] != 999999:
                    return model.v_slp_ha[q,s,z,l] == slp_area[l]
                else:
                    return pe.Constraint.Skip
            model.con_slp_area_bound = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_types, model.s_lmus, rule=slp_area_bound,
                                                    doc='bound for the area of salt land pasture on each lmu')


        ##bound on slp consumption
        if sb_upbound_inc:
            ###set the bound
            sb_max_consumption_p6 = fun.f_sa(np.array([999999],dtype=float), sen.sav['bnd_sb_consumption_p6'], 5) #999999 is arbitrary default value which mean skip constraint
            ###ravel and zip bound and dict
            sb_max_consumption_p6 = dict(zip(model.s_feed_periods, sb_max_consumption_p6))

            def sb_upper_bound(model, q, s, z, p6, l):
                if pe.value(model.p_wyear_inc_qs[q, s]) and sb_max_consumption_p6[p6] != 999999:
                    return sum(model.v_tonnes_sb_consumed[q,s,z,p6,f,l] for f in model.s_feed_pools) <= sb_max_consumption_p6[p6]
                else:
                    return pe.Constraint.Skip
            model.con_sb_upper_bound = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_types, model.s_feed_periods,
                                                     model.s_lmus, rule=sb_upper_bound, doc='upper bound for livestock sb consumption')


        ##bound on livestock supplementary feed.
        if sup_lobound_inc:
            def sup_lo_bound(model, q, s, z):
                if pe.value(model.p_wyear_inc_qs[q, s]):
                    return sum(model.v_sup_con[q,s,z,k3,g,f,p6] for k3 in model.s_supp_feeds for g in model.s_grain_pools for f in model.s_feed_pools
                    for p6 in model.s_feed_periods) >= 115
                else:
                    return pe.Constraint.Skip
            model.con_sup_lo_bound = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_types, rule=sup_lo_bound, doc='lo bound for livestock sup feed')

        ##bound on supplement per dse
        if sup_per_dse_bnd_inc:
            ### set bound
            p_sup_per_dse_bnd = sen.sav['bnd_sup_per_dse']
            ###param - propn of each fp used in the SR
            ###constraint
            l_p7 = list(model.s_season_periods)
            def sup_per_dse_bound(model, q, s):
                if pe.value(model.p_wyear_inc_qs[q, s]):
                    total_sup = sum(model.v_sup_con[q,s,z,k3,g,f,p6]
                                    * model.p_a_p6_p7[p7,p6,z] * model.p_season_seq_prob_qszp7[q,s,z,p7]
                                    for k3 in model.s_supp_feeds for g in model.s_grain_pools for f in model.s_feed_pools
                                    for p6 in model.s_feed_periods for p7 in model.s_season_periods for z in model.s_season_types)
                    wg_dse = sum((sum(model.v_sire[q,s,g0] * model.p_dse_sire[p6,z,g0] for g0 in model.s_groups_sire if pe.value(model.p_dse_sire[p6,z,g0])!=0)
                             + sum(sum(model.v_dams[q,s,k2,t1,v1,a,n1,w1,z,i,y1,g1] * model.p_dse_dams[k2,p6,t1,v1,a,n1,w1,z,i,y1,g1]
                                       for k2 in model.s_k2_birth_dams for t1 in model.s_sale_dams for v1 in model.s_dvp_dams for n1 in model.s_nut_dams
                                       for w1 in model.s_lw_dams for y1 in model.s_gen_merit_dams for g1 in model.s_groups_dams
                                       if pe.value(model.p_dse_dams[k2,p6,t1,v1,a,n1,w1,z,i,y1,g1])!=0)
                                  + sum(model.v_offs[q,s,k3,k5,t3,v3,n3,w3,z,i,a,x,y3,g3] * model.p_dse_offs[k3,k5,p6,t3,v3,n3,w3,z,i,a,x,y3,g3]
                                        for k3 in model.s_k3_damage_offs for k5 in model.s_k5_birth_offs for t3 in model.s_sale_offs for v3 in model.s_dvp_offs
                                        for n3 in model.s_nut_offs for w3 in model.s_lw_offs for x in model.s_gender for y3 in model.s_gen_merit_offs for g3 in model.s_groups_offs
                                        if pe.value(model.p_dse_offs[k3,k5,p6,t3,v3,n3,w3,z,i,a,x,y3,g3])!=0)
                                 for a in model.s_wean_times for i in model.s_tol))
                                 * model.p_wg_propn_p6z[p6, z] * model.p_a_p6_p7[p7,p6,z] * model.p_season_seq_prob_qszp7[q,s,z,p7]
                                 for p6 in model.s_feed_periods for p7 in model.s_season_periods for z in model.s_season_types)
                    return wg_dse * p_sup_per_dse_bnd == total_sup * 1000
                else:
                    return pe.Constraint.Skip
            model.con_sup_per_dse_bound = pe.Constraint(model.s_sequence_year, model.s_sequence, rule=sup_per_dse_bound,
                                                doc='total supplement fed per dse for the whole year')

        ##bnd on crop grazing intensity (kg/crop ha that can be grazed)
        if crop_grazing_intensity_bnd_inc:
            ###tonnes of crop consumed per grazable hectare
            tonnes_crop_consume_ha = sen.sav['bnd_crop_grazing_intensity']/1000 #div 1000 because input is in kgs
            l_p7 = list(model.s_season_periods)
            def f_crop_grazing_intensity(model, q, s, p7, z):
                if p7 == l_p7[-1] and pe.value(model.p_wyear_inc_qs[q, s]):
                    return (sum(sum(model.v_phase_area[q, s, p7, z, r, l] * model.p_landuse_area[r, k1] for r in model.s_phases)
                            * model.p_cropgrazing_can_occur_kl[k1,l] * tonnes_crop_consume_ha
                            for l in model.s_lmus for k1 in model.s_crops)
                            == sum(model.v_tonnes_crop_consumed[q,s,f,k1,p6,p5,z,l] for f in model.s_feed_pools
                                   for p6 in model.s_feed_periods for p5 in model.s_labperiods
                                   for l in model.s_lmus for k1 in model.s_crops
                                   if model.p_crop_DM_required[k1,p6,p5,z]!=0)) #skip if grazing doesnt occur
            model.con_crop_grazing_intensity = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_periods,
                                                             model.s_season_types, rule=f_crop_grazing_intensity,
                                                             doc='crop consumed per hectare of grazable crop')


        ##bnd numbers of dams at prejoining
        if total_dams_eqbound_inc:
            '''
            Equals to bound on total dams numbers at prejoining.
            
            '''
            prejoin_v = list(params['stock']['p_prejoin_v_dams'])[1:]  # remove the start dvp which is not a real prejoin dvp (it just has type condense).

            ##set bound using SAV
            bnd_total_dams = sen.sav['bnd_total_dams']

            ###constraint
            def f_total_dams_eqbound(model):
                return sum(model.v_dams[q,s,k2,t,v,a,n,w8,z,i,y,g1] * model.p_season_prob_qsz[q,s,z]
                           for q in model.s_sequence_year for s in model.s_sequence for k2 in model.s_k2_birth_dams
                           for t in model.s_sale_dams for v in model.s_dvp_dams for a in model.s_wean_times
                           for n in model.s_nut_dams for w8 in model.s_lw_dams for z in model.s_season_types
                           for i in model.s_tol for y in model.s_gen_merit_dams for g1 in model.s_groups_dams
                           if pe.value(model.p_mask_dams[k2,t,v,w8,z,g1]) == 1 and v in prejoin_v #only sum the numbers once per year (at prejoining)
                           ) == bnd_total_dams
            model.con_total_dams_eqbound = pe.Constraint(rule=f_total_dams_eqbound, doc='total number of dams')

        ##dam lo bound. (the sheep in a given yr equal total for all dvp divided by the number of dvps in 1 yr)
        if dams_lobound_inc:
            '''
            Lower bound on the number of dams.
            
            The constraint sets can be changed for different analysis (this is preferred rather than creating 
            a new lobound because that keeps this module smaller and easier to navigate etc).
            Typically set using SAV however for quick and dirty debugging the code below can be uncommented.
            
            Process to add/remove constraints sets:
            
                a) add/remove the set from the constraint below (will also need to add/remove from the sum)
                b) update empty array initialisation in sensitivity.py
                c) update param creation in sgen (param index will need the set added/removed)
            '''
            ##set bound using SAV
            model.p_dams_lobound = pe.Param(model.s_sale_dams, model.s_dvp_dams, model.s_season_types, model.s_groups_dams,
                                            default=0, initialize=params['stock']['p_dams_lobound'])

            ##if True this puts the lo_bnd across each starting weight - this is used in the fs optimisation to ensure each starting w has numbers and gets an optimum fs.
            ## This can make the model infeasible.
            if sen.sav['lobnd_across_startw']:
                model.s_startw_dams = pe.Set(initialize=params['stock']['startw_idx_dams'], doc='Standard LW patterns dams')
                model.p_dams_w_is_startw_ws = pe.Param(model.s_lw_dams, model.s_startw_dams, default=0, initialize=params['stock']['p_dams_w_is_startw_ws'])
            else:
                model.s_startw_dams = pe.Set(initialize=['all_w'], doc='Standard LW patterns dams')
                model.p_dams_w_is_startw_ws = pe.Param(model.s_lw_dams, model.s_startw_dams, default=1, initialize=1)

            ##manual set the bound - usually done using SAV but this is just a quick and dirty method for debugging
            # arrays = [model.s_sale_dams, model.s_dvp_dams, model.s_lw_dams, model.s_season_types, model.s_groups_dams]   #more sets can be added here to customise the bound
            # index_tvwzg = fun.cartesian_product_simple_transpose(arrays)
            # dams_lobound_tvwzg = np.zeros((len(model.s_sale_dams), len(model.s_dvp_dams), len(model.s_lw_dams), len(model.s_season_types), len(model.s_groups_dams)))
            # dams_lobound_tvwzg[-1, 4:14, :, 0, -1] = 50  #min of 50 bbt in t3 in w[0]
            # dams_lobound_tvwzg[-1, 0,0,:, 0] = 758 #min of 50 bbt in t3
            # dams_lobound_tvwzg[-1, 0,1,:, 0] = 758  #min of 50 bbt in t3
            # dams_lobound_tvwzg[-1, 0,2,:, 0] = 7.8 #min of 50 bbt in t3
            # dams_lobound = dams_lobound_tvwzg.ravel()
            # tup_tvwzg = tuple(map(tuple, index_tvwzg))
            # dams_lobound = dict(zip(tup_tvwzg, dams_lobound))

            ###constraint
            def f_dam_lobound(model, q, s, k2, t, v, ws, z, g1):
                if (pe.value(model.p_wyear_inc_qs[q, s]) and model.p_dams_lobound[t,v,z,g1]!=0
                    and any(model.p_mask_dams[k2,t,v,w8,z,g1] != 0 for w8 in model.s_lw_dams)):
                    return sum(model.v_dams[q,s,k2,t,v,a,n,w8,z,i,y,g1]
                               for a in model.s_wean_times for n in model.s_nut_dams for w8 in model.s_lw_dams
                               for i in model.s_tol for y in model.s_gen_merit_dams
                               if pe.value(model.p_mask_dams[k2,t,v,w8,z,g1]) == 1 and pe.value(model.p_dams_w_is_startw_ws[w8,ws])==1
                               ) >= model.p_dams_lobound[t,v,z,g1]
                else:
                    return pe.Constraint.Skip
            model.con_dams_lobound = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_k2_birth_dams, model.s_sale_dams
                                                   , model.s_dvp_dams, model.s_startw_dams, model.s_season_types, model.s_groups_dams
                                                   , rule=f_dam_lobound, doc='min number of dams')

        ##dams upper bound
        if dams_upbound_inc:
            '''
            Upper bound on the number of dams.

            The constraint sets can be changed for different analysis (this is preferred rather than creating 
            a new lobound because that keeps this module smaller and easier to navigate etc).
            Typically set using SAV however for quick and dirty debugging the code below can be uncommented.

            Process to add/remove constraints sets:

                a) add/remove the set from the constraint below (will also need to add/remove from the sum)
                b) update empty array initialisation in sensitivity.py
                c) update param creation in sgen (param index will need the set added/removed)
            '''
            ##set bound using SAV
            model.p_dams_upbound = pe.Param(model.s_k2_birth_dams, model.s_sale_dams, model.s_dvp_dams, model.s_season_types, model.s_groups_dams,
                                            default=0, initialize=params['stock']['p_dams_upbound'])

            ##manual set the bound - usually done using SAV but this is just a quick and dirty method for debugging
            # arrays = [model.s_sale_dams, model.s_dvp_dams, model.s_season_types, model.s_groups_dams]   #more sets can be added here to customise the bound
            # index_tvzg = fun.cartesian_product_simple_transpose(arrays)
            # dams_upbound_tvzg = np.full((len(model.s_sale_dams), len(model.s_dvp_dams), len(model.s_season_types), len(model.s_groups_dams)), np.inf)
            # dams_upbound_tvzg[0:1, 0:14,:,:] = 0  #no dam sales before dvp14 (except in DVP3 - after hgt shearing)
            # # dams_upbound_tvzg[0:1, 3:4,:,:] = np.inf   #allow sale after shearing t[0] for dams dvp3
            # dams_upbound = dams_upbound_tvzg.ravel()
            # tup_tvzg = tuple(map(tuple, index_tvzg))
            # dams_upbound = dict(zip(tup_tvzg, dams_upbound))

            ###constraint
            def f_dam_upbound(model, q, s, k2, t, v, z, g1):
                if (pe.value(model.p_wyear_inc_qs[q, s]) and model.p_dams_upbound[k2,t,v,z,g1]!=np.inf
                    and any(model.p_mask_dams[k2,t,v,w8,z,g1] != 0 for w8 in model.s_lw_dams)):
                    return sum(model.v_dams[q,s,k2,t,v,a,n,w8,z,i,y,g1]
                               for a in model.s_wean_times for n in model.s_nut_dams for w8 in model.s_lw_dams
                               for i in model.s_tol for y in model.s_gen_merit_dams
                               if pe.value(model.p_mask_dams[k2,t,v,w8,z,g1]) == 1 #if removes the masked out dams so they don't show up in .lp output.
                               ) <= model.p_dams_upbound[k2,t,v,z,g1]
                else:
                    return pe.Constraint.Skip
            model.con_dams_upbound = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_k2_birth_dams, model.s_sale_dams
                                                   , model.s_dvp_dams, model.s_season_types, model.s_groups_dams
                                                   , rule=f_dam_upbound, doc='max number of dams_tv')

        ##offs lo bound
        if offs_lobound_inc:
            '''
            Lower bound on the number of offs.

            The constraint sets can be changed for different analysis (this is preferred rather than creating 
            a new lobound because that keeps this module smaller and easier to navigate etc).
            Typically set using SAV however for quick and dirty debugging the code below can be uncommented.

            Process to add/remove constraints sets:

                a) add/remove the set from the constraint below (will also need to add/remove from the sum)
                b) update empty array initialisation in sensitivity.py
                c) update param creation in sgen (param index will need the set added/removed)
            '''
            ##set bound using SAV
            model.p_offs_lobound = pe.Param(model.s_k3_damage_offs, model.s_sale_offs, model.s_dvp_offs, model.s_season_types, model.s_gender,
                                            model.s_groups_offs, default=0, initialize=params['stock']['p_offs_lobound'])

            ##if True this puts the lo_bnd across each starting weight - this is used in the fs optimisation to ensure each starting w has numbers and gets an optimum fs.
            ## This can make the model infeasible.
            if sen.sav['lobnd_across_startw']:
                model.s_startw_offs = pe.Set(initialize=params['stock']['startw_idx_offs'], doc='Standard LW patterns offs')
                model.p_offs_w_is_startw_ws = pe.Param(model.s_lw_offs, model.s_startw_offs, default=0, initialize=params['stock']['p_offs_w_is_startw_ws'])
            else:
                model.s_startw_offs = pe.Set(initialize=['all_w'], doc='Standard LW patterns offs')
                model.p_offs_w_is_startw_ws = pe.Param(model.s_lw_offs, model.s_startw_offs, default=1, initialize=1)

            ##manual set the bound - usually done using SAV but this is just a quick and dirty method for debugging
            ###keys to build arrays for the specified slices
            # arrays = [model.s_sale_offs, model.s_dvp_offs, model.s_lw_offs,model.s_season_types, model.s_gender, model.s_groups_offs]   #more sets can be added here to customise the bound
            # index_tvwzxg = fun.cartesian_product_simple_transpose(arrays)
            # ###build array for the axes of the specified slices
            # offs_lowbound_tvwzxg = np.zeros((len(model.s_sale_offs), len(model.s_dvp_offs), len(model.s_lw_offs), len(model.s_season_types), len(model.s_gender), len(model.s_groups_offs)))
            # ###set the bound
            # offs_lowbound_tvwzxg[0, 0,0,:, 0,0] = 50
            # offs_lowbound_tvwzxg[0, 0,1,:, 0,0] = 50
            # offs_lowbound_tvwzxg[0, 0,2,:, 0,0] = 0
            # ###ravel and zip bound and dict
            # offs_lowbound = offs_lowbound_tvwzxg.ravel()
            # tup_tvwzxg = tuple(map(tuple, index_tvwzxg))
            # offs_lowbound = dict(zip(tup_tvwzxg, offs_lowbound))

            ###constraint
            def f_off_lobound(model, q, s, k3, k5, t, v, ws, z, x, g3):
                if (pe.value(model.p_wyear_inc_qs[q, s]) and model.p_offs_lobound[k3,t,v,z,x,g3]!=0\
                        and any(model.p_mask_offs[k3,v,w8,z,x,g3] != 0 for w8 in model.s_lw_offs)):
                    return sum(model.v_offs[q,s,k3,k5,t,v,n3,w8,z,i,a,x,y3,g3]
                               for a in model.s_wean_times for n3 in model.s_nut_offs
                               for w8 in model.s_lw_offs for i in model.s_tol for y3 in model.s_gen_merit_offs
                               if pe.value(model.p_mask_offs[k3,v,w8,z,x,g3]) == 1 and pe.value(model.p_offs_w_is_startw_ws[w8,ws])==1
                               ) >= model.p_offs_lobound[k3,t,v,z,x,g3]
                else:
                    return pe.Constraint.Skip
            model.con_offs_lobound = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_k3_damage_offs
                                                   , model.s_k5_birth_offs, model.s_sale_offs, model.s_dvp_offs
                                                   , model.s_startw_offs, model.s_season_types, model.s_gender, model.s_groups_offs
                                                   , rule=f_off_lobound, doc='min number of offs')

        ##offs upper bound
        if offs_upbound_inc:
            '''
            Upper bound on the number of offs.

            The constraint sets can be changed for different analysis (this is preferred rather than creating 
            a new upbound because that keeps this module smaller and easier to navigate etc).
            Typically set using SAV however for quick and dirty debugging the code below can be uncommented.

            Process to add/remove constraints sets:

                a) add/remove the set from the constraint below (will also need to add/remove from the sum)
                b) update empty array initialisation in sensitivity.py
                c) update param creation in sgen (param index will need the set added/removed)
            '''
            ##set bound using SAV
            model.p_offs_upbound = pe.Param(model.s_k3_damage_offs, model.s_sale_offs, model.s_dvp_offs, model.s_season_types, model.s_gender,
                                            model.s_groups_offs, default=0, initialize=params['stock']['p_offs_upbound'])

            ##manual set the bound - usually done using SAV but this is just a quick and dirty method for debugging
            ###keys to build arrays for the specified slices
            # arrays = [model.s_sale_offs, model.s_dvp_offs, model.s_lw_offs,model.s_season_types, model.s_gender, model.s_groups_offs]   #more sets can be added here to customise the bound
            # index_tvwzxg = fun.cartesian_product_simple_transpose(arrays)
            # ###build array for the axes of the specified slices
            # offs_upbound_tvwzxg = np.zeros((len(model.s_sale_offs), len(model.s_dvp_offs), len(model.s_lw_offs), len(model.s_season_types), len(model.s_gender), len(model.s_groups_offs)))
            # ###set the bound
            # offs_upbound_tvwzxg[0, 0,0,:, 0,0] = 50
            # offs_upbound_tvwzxg[0, 0,1,:, 0,0] = 50
            # offs_upbound_tvwzxg[0, 0,2,:, 0,0] = 0
            # ###ravel and zip bound and dict
            # offs_upbound = offs_lowbound_tvwzxg.ravel()
            # tup_tvwzxg = tuple(map(tuple, index_tvwzxg))
            # offs_upbound = dict(zip(tup_tvwzxg, offs_upbound))

            ###constraint
            def f_off_upbound(model, q, s, k3, t, v, z, x, g3):
                if pe.value(model.p_wyear_inc_qs[q, s]) and model.p_offs_upbound[k3,t,v,z,x,g3]<999999:
                    return sum(model.v_offs[q,s,k3,k5,t,v,n3,w8,z,i,a,x,y3,g3]
                               for k5 in model.s_k5_birth_offs for a in model.s_wean_times for n3 in model.s_nut_offs
                               for w8 in model.s_lw_offs for i in model.s_tol for y3 in model.s_gen_merit_offs
                               ) <= model.p_offs_upbound[k3,t,v,z,x,g3]
                else:
                    return pe.Constraint.Skip
            model.con_offs_upbound = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_k3_damage_offs, model.s_sale_offs
                                                   , model.s_dvp_offs, model.s_season_types, model.s_gender, model.s_groups_offs
                                                   , rule=f_off_upbound, doc='max number of offs')


        ##prog upper bound
        if prog_upbound_inc:
            '''
            Upper bound prog.

            The constraint sets can be changed for different analysis (this is preferred rather than creating 
            a new upbound because that keeps this module smaller and easier to navigate etc).

            Process to add/remove constraints sets:

                a) add/remove the set from the constraint below (will also need to add/remove from the sum)
                b) update empty array initialisation in sensitivity.py
                c) update param creation in sgen (param index will need the set added/removed)
            '''
            ##set bound using SAV
            model.p_prog_upbound = pe.Param(model.s_k3_damage_offs, model.s_sale_prog, model.s_gender,
                                            model.s_groups_prog, default=0, initialize=params['stock']['p_prog_upbound'])

            ###constraint
            def f_prog_upbound(model, q, s, k3, t, x, g2):
                if pe.value(model.p_wyear_inc_qs[q, s]) and model.p_prog_upbound[k3,t,x,g2]<999999:
                    return sum(model.v_prog[q,s,k3,k5,t,w,z,i,a,x,g2]
                               for k5 in model.s_k5_birth_offs for a in model.s_wean_times
                               for w in model.s_lw_prog for z in model.s_season_types for i in model.s_tol
                               ) <= model.p_prog_upbound[k3,t,x,g2]
                else:
                    return pe.Constraint.Skip
            model.con_prog_upbound = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_k3_damage_offs
                                                   , model.s_sale_prog, model.s_gender, model.s_groups_prog
                                                   , rule=f_prog_upbound, doc='max number of prog')


        ##total dams scanned. Sums mated dams in all scanning dvps
        ###build bound if turned on
        if total_dams_scanned_bound_inc:
            ###set the bound
            total_dams_scanned = fun.f_sa(999999, sen.sav['bnd_total_dams_scanned'], 5) #999999 is arbitrary default value which mean skip constraint
            ###scan dvps
            scan_v = list(params['stock']['p_scan_v_dams'])
            ###constraint - sum all mated dams in scan dvp.
            def f_total_dams_scanned(model,q,s,z):
                if pe.value(model.p_wyear_inc_qs[q, s]) and total_dams_scanned != 999999:
                    return sum(model.v_dams[q,s,k28,t,v,a,n,w8,z,i,y,g1] for k28 in model.s_k2_birth_dams for t in model.s_sale_dams
                               for v in model.s_dvp_dams for a in model.s_wean_times for n in model.s_nut_dams for w8 in model.s_lw_dams
                               for i in model.s_tol for y in model.s_gen_merit_dams for g1 in model.s_groups_dams
                               if pe.value(model.p_mask_dams[k28,t,v,w8,z,g1]) == 1 and v in scan_v and k28 != 'NM-0') \
                       == total_dams_scanned
                else:
                    pe.Constraint.Skip
            model.con_total_dams_scanned = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_types, rule=f_total_dams_scanned, doc='total dams scanned')

        ##force 5yo dam retention - fix a proportion of dams at 6yo scanning dvp.
        ###build bound if turned on
        if force_5yo_retention_inc:
            ###set the bound
            propn_dams_retained = fun.f_sa(999, sen.sav['bnd_propn_dam5_retained'], 5) #999 is arbitrary default value which mean skip constraint
            ###5yr scan dvp
            scan5_v = list(params['stock']['p_scan_v_dams'])[4]
            ###6yr scan dvp
            scan6_v = list(params['stock']['p_scan_v_dams'])[5]
            ###constraint - sum all mated dams in scan dvp.
            def retention_5yo_dams(model,q,s,z):
                if (pe.value(model.p_wyear_inc_qs[q, s]) and propn_dams_retained != 999
                        and any(model.p_mask_dams[k2,t,scan5_v,w8,z,g1] != 0 for k2 in model.s_k2_birth_dams for w8 in model.s_lw_dams
                                for t in model.s_sale_dams for g1 in model.s_groups_dams)):
                    return (propn_dams_retained) * sum(model.v_dams[q,s,k28,t,v,a,n,w8,z,i,y,g1] for k28 in model.s_k2_birth_dams
                                                       for t in model.s_sale_dams for v in model.s_dvp_dams
                                                       for a in model.s_wean_times for n in model.s_nut_dams for w8 in model.s_lw_dams
                                                       for i in model.s_tol for y in model.s_gen_merit_dams for g1 in model.s_groups_dams
                                                       if pe.value(model.p_mask_dams[k28,t,v,w8,z,g1]) == 1 and v in scan5_v) \
                       == sum(model.v_dams[q,s,k28,t,v,a,n,w8,z,i,y,g1] for k28 in model.s_k2_birth_dams
                              for t in model.s_sale_dams for v in model.s_dvp_dams for a in model.s_wean_times
                              for n in model.s_nut_dams for w8 in model.s_lw_dams for i in model.s_tol
                              for y in model.s_gen_merit_dams for g1 in model.s_groups_dams
                              if pe.value(model.p_mask_dams[k28,t,v,w8,z,g1]) == 1 and v in scan6_v)
                else:
                    return pe.Constraint.Skip
            model.con_retention_5yo_dams = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_types, rule=retention_5yo_dams, doc='force retention of 5yo dams')

        ##bound to fix the proportion of dams being mated - Proportion of mated dams relative to total dams, optimised across the w axis
        ###this bound does not count the number of females that are transferred to offs.
        ###build bound if turned on
        if bnd_propn_dams_mated_inc:
            ###build param - inf values are skipped in the constraint building so inf means the model can optimise the propn mated
            model.p_prop_dams_mated = pe.Param(model.s_dvp_dams, model.s_season_types, model.s_groups_dams
                                               , default=0, initialize=params['stock']['p_prop_dams_mated'])
            ###constraint
            #todo add an i axis to the constraint
            def f_propn_dams_mated(model, q, s, v, z, g1):
                if (model.p_prop_dams_mated[v,z,g1]==np.inf or not pe.value(model.p_wyear_inc_qs[q, s]) or
                        all(model.p_mask_dams[k2,t,v,w8,z,g1] == 0
                            for k2 in model.s_k2_birth_dams for t in model.s_sale_dams for w8 in model.s_lw_dams)):
                    return pe.Constraint.Skip
                else:
                    return sum(model.v_dams[q,s,'NM-0',t,v,a,n,w8,z,i,y,g1] for t in model.s_sale_dams
                               for a in model.s_wean_times for n in model.s_nut_dams for w8 in model.s_lw_dams
                               for i in model.s_tol for y in model.s_gen_merit_dams
                               if pe.value(model.p_mask_dams['NM-0',t,v,w8,z,g1]) == 1
                               ) == sum(model.v_dams[q,s,k2,t,v,a,n,w8,z,i,y,g1] for k2 in model.s_k2_birth_dams for t in model.s_sale_dams
                               for a in model.s_wean_times for n in model.s_nut_dams for w8 in model.s_lw_dams
                               for i in model.s_tol for y in model.s_gen_merit_dams
                               if pe.value(model.p_mask_dams[k2,t,v,w8,z,g1]) == 1) * (1 - model.p_prop_dams_mated[v,z,g1])
            model.con_propn_dams_mated = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_dvp_dams, model.s_season_types, model.s_groups_dams, rule=f_propn_dams_mated,
                                                       doc='proportion of dams mated')

        ##bound to fix the proportion of dams being mated - Proportion fixed optimised across the w axis
        ###build bound if turned on
        if bnd_propn_dams_mated_w_inc:
            ###build param - inf values are skipped in the constraint building so inf means the model can optimise the propn mated
            model.p_prop_dams_mated = pe.Param(model.s_dvp_dams, model.s_season_types, model.s_groups_dams
                                               , default=0, initialize=params['stock']['p_prop_dams_mated'])
            ###constraint
            #todo add an i axis to the constraint
            def f_propn_dams_mated_w(model, q, s, v, w8, z, g1):
                if (model.p_prop_dams_mated[v,z,g1]==np.inf or not pe.value(model.p_wyear_inc_qs[q, s]) or
                        all(model.p_mask_dams[k2,t,v,w8,z,g1] == 0
                            for k2 in model.s_k2_birth_dams for t in model.s_sale_dams)):
                    return pe.Constraint.Skip
                else:
                    return sum(model.v_dams[q,s,'NM-0',t,v,a,n,w8,z,i,y,g1] for t in model.s_sale_dams
                               for a in model.s_wean_times for n in model.s_nut_dams
                               for i in model.s_tol for y in model.s_gen_merit_dams
                               if pe.value(model.p_mask_dams['NM-0',t,v,w8,z,g1]) == 1
                               ) == sum(model.v_dams[q,s,k2,t,v,a,n,w8,z,i,y,g1] for k2 in model.s_k2_birth_dams
                               for t in model.s_sale_dams for a in model.s_wean_times for n in model.s_nut_dams
                               for i in model.s_tol for y in model.s_gen_merit_dams
                               if pe.value(model.p_mask_dams[k2,t,v,w8,z,g1]) == 1) * (1 - model.p_prop_dams_mated[v,z,g1])
            model.con_propn_dams_mated_w = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_dvp_dams
                                                , model.s_lw_dams, model.s_season_types, model.s_groups_dams
                                                , rule=f_propn_dams_mated_w, doc='proportion of dams mated with w set')

        ##bound to fix the proportion of twice dry dams sold. Proportion of twice dry dams is an input in uinp ce[2, ....].
        ###build bound if turned on
        if bnd_sale_twice_drys_inc:
            '''Constraint forces x percent of the dry dams to be sold (all the twice drys). They can be sold in either 
             sale op (after scanning or at shearing if shearing occurs before the next prejoining).
             Thus the drys just have to be sold sometime between scanning and the following prejoining.
             
             The assumption is that the twice dry propn is the same for all w slices and thus a propn of the drys 
             must be sold from each w slice.
             
             This constraint is equal to and really says only retain drys that are not twice dry.'''
            ###build param - inf values are skipped in the constraint building so inf means the model can optimise the propn mated
            model.p_prop_twice_dry_dams = pe.Param(model.s_dvp_dams, model.s_season_types, model.s_tol, model.s_gen_merit_dams,
                                                   model.s_groups_dams, default=0, initialize=params['stock']['p_prop_twice_dry_dams'])

            l_v1 = list(model.s_dvp_dams)
            scan_v = list(params['stock']['p_scan_v_dams'])
            prejoin_v = list(params['stock']['p_prejoin_v_dams'])[1:] #remove the start dvp which is not a real prejoin dvp (it just has type condense).
            next_prejoin_v = prejoin_v[1:] #dvp before following prejoining

            ###constraint
            def f_propn_drys_sold(model, q, s, v, w, z, i, y, g1):
                '''Force the model so that the only drys that can be retain are not twice dry (essentially forcing the sale of twice drys)'''
                if (pe.value(model.p_wyear_inc_qs[q, s]) and v in scan_v[:-1] and model.p_prop_twice_dry_dams[v,z,i,y,g1]!=0
                        and any(model.p_mask_dams['00-0',t,v,w,z,g1]==1 for t in model.s_sale_dams)): #use 00 numbers at scanning. Don't want to include the last prejoining dvp because there is no sale limit in the last year.
                    idx_scan = scan_v.index(v) #which prejoining is the current v
                    idx_v_next_prejoin = next_prejoin_v[idx_scan] #all the twice drys must be sold by the following prejoining
                    v_sale = l_v1[l_v1.index(idx_v_next_prejoin) - 1]
                    return sum(model.v_dams[q,s,'00-0','t2',v_sale,a,n,w,z,i,y,g1]
                               for a in model.s_wean_times for n in model.s_nut_dams
                               if pe.value(model.p_mask_dams['00-0','t2',v_sale,w,z,g1]) == 1
                               ) == sum(model.v_dams[q,s,'00-0',t,v,a,n,w,z,i,y,g1] for t in model.s_sale_dams
                                        for a in model.s_wean_times for n in model.s_nut_dams
                                        if pe.value(model.p_mask_dams['00-0',t,v,w,z,g1]) == 1
                                        ) * (1-model.p_prop_twice_dry_dams[v,z,i,y,g1])
                else:
                    return pe.Constraint.Skip
            model.con_propn_drys_sold = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_dvp_dams
                                                      , model.s_lw_dams, model.s_season_types, model.s_tol
                                                      , model.s_gen_merit_dams, model.s_groups_dams, rule=f_propn_drys_sold
                                                      , doc='proportion of dry dams sold each year')

        ##bound to fix a proportion of single (& twin) ewes to be sold. To represent the sale of dry ewes without
        ### having to uncluster the b axis. The extra proportion of dams to sell (above empty ewes) is a sav[p_min_prop_single-dams_sold].
        ###build bound if turned on
        if bnd_propn_singles_inc:
            '''Constraint forces at least x percent of the single/twin dams to be sold. They have to be sold  
             before the next prejoining.
             The idea is to represent selling ewes that fail to rear a lamb without having to uncluster the b axis.

             The assumption is that the dams that GBAL is the same across the w slices because the same propn  
             must be sold from each w slice.

             This constraint is representing retaining dams that didn't GBAL, except that difference in post weaning LWC aren't represented in the sale animals.'''
            ###build param - inf values are skipped in the constraint building so inf means the model can optimise the propn mated
            model.p_min_prop_single_dams_sold = pe.Param(model.s_dvp_dams, model.s_season_types, model.s_tol,
                                                   model.s_gen_merit_dams,
                                                   model.s_groups_dams, default=0,
                                                   initialize=params['stock']['p_min_prop_single_dams_sold'])

            l_v1 = list(model.s_dvp_dams)
            scan_v = list(params['stock']['p_scan_v_dams'])
            prejoin_v = list(params['stock']['p_prejoin_v_dams'])[1:]  #remove the start dvp, it is not true pre-join.
            next_prejoin_v = prejoin_v[1:]  #dvp before following prejoining

            ###constraint
            def f_propn_singles_sold(model, q, s, v, w, z, i, y, g1):
                '''A maximum proportion of single ewes can be retained, hence forcing sale of the remainder'''
                if (pe.value(model.p_wyear_inc_qs[q, s]) and v in scan_v[:-1] and model.p_min_prop_single_dams_sold[v, z, i, y, g1] != 0
                        and any(model.p_mask_dams['11-0', t, v, w, z, g1] == 1 for t in model.s_sale_dams)):  #use 11 numbers at scanning. Don't want to include the last prejoining dvp because there is no sale limit in the last year.
                    idx_scan = scan_v.index(v)  #which prejoining is the current v
                    idx_v_next_prejoin = next_prejoin_v[idx_scan]  #the sale must be before the following prejoining
                    v_sale = l_v1[l_v1.index(idx_v_next_prejoin) - 1]
                    return sum(model.v_dams[q, s, '11-0', 't2', v_sale, a, n, w, z, i, y, g1]
                               for a in model.s_wean_times for n in model.s_nut_dams
                               if pe.value(model.p_mask_dams['11-0', 't2', v_sale, w, z, g1]) == 1
                               ) <= sum(model.v_dams[q, s, '11-0', t, v, a, n, w, z, i, y, g1] for t in model.s_sale_dams
                        for a in model.s_wean_times for n in model.s_nut_dams
                        if pe.value(model.p_mask_dams['11-0', t, v, w, z, g1]) == 1
                        ) * (1 - model.p_min_prop_single_dams_sold[v, z, i, y, g1])
                else:
                    return pe.Constraint.Skip

            model.con_propn_singles_sold = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_dvp_dams
                                                      , model.s_lw_dams, model.s_season_types, model.s_tol
                                                      , model.s_gen_merit_dams, model.s_groups_dams, rule=f_propn_singles_sold
                                                      , doc='proportion of single dams sold each year')


        ##bound to fix a proportion of single (& twin) ewes to be sold. To represent the sale of dry ewes without
        ### having to uncluster the b axis. The extra proportion of dams to sell (above empty ewes) is a sav[p_min_prop_single-dams_sold].
        ###build bound if turned on
        if bnd_propn_twins_inc:
            '''Constraint forces at least x percent of the single/twin dams to be sold. They have to be sold  
             before the next prejoining.
             The idea is to represent selling ewes that fail to rear a lamb without having to uncluster the b axis.

             The assumption is that the dams that GBAL is the same across the w slices because the same propn  
             must be sold from each w slice.

             This constraint is representing retaining dams that didn't GBAL, except that difference in post weaning LWC aren't represented in the sale animals.'''
            ###build param - inf values are skipped in the constraint building so inf means the model can optimise the propn mated
            model.p_min_prop_twin_dams_sold = pe.Param(model.s_dvp_dams, model.s_season_types, model.s_tol,
                                                   model.s_gen_merit_dams,
                                                   model.s_groups_dams, default=0,
                                                   initialize=params['stock']['p_min_prop_twin_dams_sold'])

            l_v1 = list(model.s_dvp_dams)
            scan_v = list(params['stock']['p_scan_v_dams'])
            prejoin_v = list(params['stock']['p_prejoin_v_dams'])[1:]  #remove the start dvp, it is not true pre-join.
            next_prejoin_v = prejoin_v[1:]  #dvp before following prejoining

            ###constraint
            def f_propn_twins_sold(model, q, s, v, w, z, i, y, g1):
                '''A maximum proportion of twin ewes can be retained, hence forcing sale of the remainder'''
                if (pe.value(model.p_wyear_inc_qs[q, s]) and v in scan_v[:-1] and model.p_min_prop_twin_dams_sold[v, z, i, y, g1] != 0
                        and any(model.p_mask_dams['22-0', t, v, w, z, g1] == 1 for t in model.s_sale_dams)):  #use 11 numbers at scanning. Don't want to include the last prejoining dvp because there is no sale limit in the last year.
                    idx_scan = scan_v.index(v)  #which prejoining is the current v
                    idx_v_next_prejoin = next_prejoin_v[idx_scan]  #the sale must be before the following prejoining
                    v_sale = l_v1[l_v1.index(idx_v_next_prejoin) - 1]
                    return sum(model.v_dams[q, s, '22-0', 't2', v_sale, a, n, w, z, i, y, g1]
                               for a in model.s_wean_times for n in model.s_nut_dams
                               if pe.value(model.p_mask_dams['22-0', 't2', v_sale, w, z, g1]) == 1
                               ) <= sum(model.v_dams[q, s, '22-0', t, v, a, n, w, z, i, y, g1] for t in model.s_sale_dams
                        for a in model.s_wean_times for n in model.s_nut_dams
                        if pe.value(model.p_mask_dams['22-0', t, v, w, z, g1]) == 1
                        ) * (1 - model.p_min_prop_twin_dams_sold[v, z, i, y, g1])
                else:
                    return pe.Constraint.Skip

            model.con_propn_twins_sold = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_dvp_dams
                                                      , model.s_lw_dams, model.s_season_types, model.s_tol
                                                      , model.s_gen_merit_dams, model.s_groups_dams, rule=f_propn_twins_sold
                                                      , doc='proportion of twin dams sold each year')

        ##bound to force the retention of drys until the dvp when other ewes are sold.
        # The bound is only for t[0] (sale at shearing) t[1] (sale at scanning) is handled in the generator.
        if bnd_dry_retained_inc:
            ###build param
            model.p_prop_dry_t0_dams = pe.Param(model.s_dvp_dams, model.s_wean_times, model.s_nut_dams, model.s_lw_dams
                                                , model.s_season_types, model.s_tol, model.s_gen_merit_dams
                                                , model.s_groups_dams, default=0, initialize=params['stock']['p_prop_dry_t0_dams'])
            model.p_drys_retained = pe.Param(model.s_dvp_dams, model.s_season_types, model.s_groups_dams
                                             ,default=0, initialize=params['stock']['p_drys_retained'])

            ###constraint
            def f_retention_drys(model, q, s, v, z, i, g1):
                '''Force the model so that the drys can only be sold when the other ewes are sold (essentially forcing the retention of drys).
                   The number of drys sold must be less than the sum of the other k2 slices'''
                #todo add birth timing to p_prop_dry_t0_dams when gbal is activated
                if pe.value(model.p_wyear_inc_qs[q, s]) and any(model.p_mask_dams['00-0','t0',v,w,z,g1]!=0 for w in model.s_lw_dams) and model.p_drys_retained[v,z,g1]!=0:
                    return sum(model.v_dams[q,s,'00-0','t0',v,a,n,w,z,i,y,g1]
                               for a in model.s_wean_times for n in model.s_nut_dams for w in model.s_lw_dams for y in model.s_gen_merit_dams
                               if pe.value(model.p_mask_dams['00-0','t0',v,w,z,g1]) == 1
                               ) <= max(model.p_prop_dry_t0_dams[v,a,n,w,z,i,y,g1] for a in model.s_wean_times for n in model.s_nut_dams  #take max to reduce size. Needs to be max so that all drys can be sold. This will allow a tiny bit of slippage (can sell more slightly more drys than the exact dry propn)
                                        for w in model.s_lw_dams for y in model.s_gen_merit_dams) * sum(model.v_dams[q,s,k2,'t0',v,a,n,w,z,i,y,g1]
                                        for k2 in model.s_k2_birth_dams for a in model.s_wean_times for n in model.s_nut_dams
                                        for w in model.s_lw_dams for y in model.s_gen_merit_dams
                                        if pe.value(model.p_mask_dams['00-0','t0',v,w,z,g1]) == 1) #sums the k2 axis except for drys.
                else:
                    return pe.Constraint.Skip

            model.con_retention_drys = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_dvp_dams, model.s_season_types, model.s_tol, model.s_groups_dams, rule=f_retention_drys,
                                                       doc='force the retention of drys until other dams are sold')

        ##SR - this can't set the sr on an actual pasture but it means different pastures provide a different level of carry capacity although nothing fixes sheep to that pasture
        ###build bound
        if sr_bound_inc:
            ###initilise
            len_q = sinp.structuralsa['i_len_q']  # number of years in MP model
            keys_q = np.array(['q%s' % i for i in range(len_q)])
            bnd_sr_qt = fun.f_sa(np.array([99999]), sen.sav['bnd_sr_Qt'][0:len_q, pinp.general['pas_inc_t']], 5)  # 99999 is arbitrary default value which mean skip constraint
            bnd_sr_qt = fun.f1_make_pyomo_dict(bnd_sr_qt, [keys_q, model.s_pastures])
            ###constraint
            l_p7 = list(model.s_season_periods)
            p7_end_gs0 = l_p7[pinp.general['i_gs_p7_end'][0]]  # p7 period from growing season 0.
            def SR_bound(model, q, s):
                if pe.value(model.p_wyear_inc_qs[q, s]) and any(bnd_sr_qt[q,t]!=99999 for t in model.s_pastures) and (sum(model.p_wg_propn_p6z[p6,z] * model.p_a_p6_p7[p7,p6,z] * model.p_season_seq_prob_qszp7[q,s,z,p7]
                                    for p6 in model.s_feed_periods for p7 in model.s_season_periods for z in model.s_season_types)>0):
                    rhs_dse = sum(model.v_phase_area[q, s, p7_end_gs0, z, r, l] * model.p_pasture_area[r, t] * bnd_sr_qt[q,t]
                                  * model.p_wg_propn_p6z[p6, z] * model.p_a_p6_p7[p7, p6, z] * model.p_season_seq_prob_qszp7[q, s, z, p7]
                                  for p6 in model.s_feed_periods for p7 in model.s_season_periods
                                  for r in model.s_phases for l in model.s_lmus for t in model.s_pastures for z in model.s_season_types)
                    dse = sum((sum(model.v_sire[q,s,g0] * model.p_dse_sire[p6,z,g0] for g0 in model.s_groups_sire if pe.value(model.p_dse_sire[p6,z,g0])!=0)
                             + sum(sum(model.v_dams[q,s,k2,t1,v1,a,n1,w1,z,i,y1,g1] * model.p_dse_dams[k2,p6,t1,v1,a,n1,w1,z,i,y1,g1]
                                       for k2 in model.s_k2_birth_dams for t1 in model.s_sale_dams for v1 in model.s_dvp_dams for n1 in model.s_nut_dams
                                       for w1 in model.s_lw_dams for y1 in model.s_gen_merit_dams for g1 in model.s_groups_dams
                                       if pe.value(model.p_dse_dams[k2,p6,t1,v1,a,n1,w1,z,i,y1,g1])!=0)
                                  + sum(model.v_offs[q,s,k3,k5,t3,v3,n3,w3,z,i,a,x,y3,g3] * model.p_dse_offs[k3,k5,p6,t3,v3,n3,w3,z,i,a,x,y3,g3]
                                        for k3 in model.s_k3_damage_offs for k5 in model.s_k5_birth_offs for t3 in model.s_sale_offs for v3 in model.s_dvp_offs
                                        for n3 in model.s_nut_offs for w3 in model.s_lw_offs for x in model.s_gender for y3 in model.s_gen_merit_offs for g3 in model.s_groups_offs
                                        if pe.value(model.p_dse_offs[k3,k5,p6,t3,v3,n3,w3,z,i,a,x,y3,g3])!=0)
                                 for a in model.s_wean_times for i in model.s_tol))
                            * model.p_wg_propn_p6z[p6,z] * model.p_a_p6_p7[p7,p6,z] * model.p_season_seq_prob_qszp7[q,s,z,p7]
                            for p6 in model.s_feed_periods for p7 in model.s_season_periods for z in model.s_season_types)
                    return dse == rhs_dse
                else:
                    return pe.Constraint.Skip
            model.con_SR_bound = pe.Constraint(model.s_sequence_year, model.s_sequence, rule=SR_bound,
                                                doc='stocking rate bound for each feed period')

        ##LW - target difference in LW compared to the base w (nut 0) at the end of p7[0]. Used in MP model.
        ###build bound
        if lw_bound_inc:
            ###initilise
            model.p_lw_diff_from_target_k2tva1nwziyg1 = pe.Param(model.s_k2_birth_dams, model.s_sale_dams,
                                        model.s_dvp_dams, model.s_wean_times, model.s_nut_dams,
                                        model.s_lw_dams, model.s_season_types, model.s_tol, model.s_gen_merit_dams,
                                        model.s_groups_dams,
                                        initialize=params['stock']['p_lw_diff_from_target_k2tva1nwziyg1'], default=0.0, mutable=False,
                                        doc='')
            model.p_lw_diff_from_target_k3k5tvnwziaxyg3 = pe.Param(model.s_k3_damage_offs, model.s_k5_birth_offs,
                                        model.s_sale_offs, model.s_dvp_offs, model.s_nut_offs, model.s_lw_offs,
                                        model.s_season_types, model.s_tol, model.s_wean_times, model.s_gender,
                                        model.s_gen_merit_offs, model.s_groups_offs,
                                        initialize=params['stock']['p_lw_diff_from_target_k3k5tvnwziaxyg3'], default=0.0, mutable=False,
                                        doc='')
            ###constraint
            def lw_dams_bound(model,q,s,k2,t1,v1,a,z,i,y1,g1):
                ##note 1: the lw_diff param is 0 unless dvp is end of node 1.
                ##note 2: constraint is only active in q[0] at the end of node 0.
                if pe.value(model.p_wyear_inc_qs[q, s]) and q=='q0' and any(pe.value(model.p_lw_diff_from_target_k2tva1nwziyg1[k2,t1,v1,a,n1,w1,z,i,y1,g1])!=0 for n1 in model.s_nut_dams for w1 in model.s_lw_dams):
                    return sum(model.v_dams[q,s,k2,t1,v1,a,n1,w1,z,i,y1,g1] * model.p_lw_diff_from_target_k2tva1nwziyg1[k2,t1,v1,a,n1,w1,z,i,y1,g1]
                               for n1 in model.s_nut_dams for w1 in model.s_lw_dams
                               if pe.value(model.p_lw_diff_from_target_k2tva1nwziyg1[k2,t1,v1,a,n1,w1,z,i,y1,g1])!=0) == 0
                else:
                    return pe.Constraint.Skip
            model.con_lw_dams_bound = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_k2_birth_dams, model.s_sale_dams,
                                                    model.s_dvp_dams, model.s_wean_times, model.s_season_types, model.s_tol, model.s_gen_merit_dams,
                                                    model.s_groups_dams, rule=lw_dams_bound,
                                                    doc='target difference in LW compared to the base w (nut 0)')

            def lw_offs_bound(model,q,s,k3,k5,t3,v3,z,i,a,x,y3,g3):
                ##note 1: the lw_diff param is 0 unless dvp is end of node 0.
                ##note 2: constraint is only active in q[0] at the end of node 0.
                if pe.value(model.p_wyear_inc_qs[q, s]) and q=='q0' and any(pe.value(model.p_lw_diff_from_target_k3k5tvnwziaxyg3[k3,k5,t3,v3,n3,w3,z,i,a,x,y3,g3])!=0 for n3 in model.s_nut_offs for w3 in model.s_lw_offs):
                    return sum(model.v_offs[q,s,k3,k5,t3,v3,n3,w3,z,i,a,x,y3,g3] * model.p_lw_diff_from_target_k3k5tvnwziaxyg3[k3,k5,t3,v3,n3,w3,z,i,a,x,y3,g3]
                               for n3 in model.s_nut_offs for w3 in model.s_lw_offs
                               if pe.value(model.p_lw_diff_from_target_k3k5tvnwziaxyg3[k3,k5,t3,v3,n3,w3,z,i,a,x,y3,g3])!=0) == 0
                else:
                    return pe.Constraint.Skip
            model.con_lw_offs_bound = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_k3_damage_offs, model.s_k5_birth_offs, model.s_sale_offs,
                                                    model.s_dvp_offs, model.s_season_types, model.s_tol, model.s_wean_times, model.s_gender, model.s_gen_merit_offs,
                                                    model.s_groups_offs, rule=lw_offs_bound,
                                                    doc='target difference in LW compared to the base w (nut 0)')

        ##landuse bound
        ###build bound if turned on
        if landuse_bound_inc:
            ###setbound using % of farm area
            area_bound_klz = fun.f_sa(np.array([99999]), sen.sav['bnd_landuse_area_klz'][pinp.all_landuse_mask_k,:,:][:,lmu_mask,:][:,:,z_mask], 5)  # 99999 is arbitrary default value which mean skip constraint
            arrays = [model.s_landuses, model.s_lmus, model.s_season_types]
            index_klz = fun.cartesian_product_simple_transpose(arrays)
            tup_klz = tuple(map(tuple, index_klz))
            area_bound_klz = dict(zip(tup_klz, area_bound_klz.ravel()))
            ###constraint
            l_p7 = list(model.s_season_periods)
            def k_bound(model, q, s, p7, k, l, z):
                if p7 == l_p7[-1] and area_bound_klz[k,l,z]!=99999 and pe.value(model.p_wyear_inc_qs[q, s]):
                    return(sum(model.v_phase_area[q,s,p7,z,r,l] * model.p_landuse_area[r, k] for r in model.s_phases)
                           == area_bound_klz[k,l,z])
                else:
                    return pe.Constraint.Skip
            model.con_landuse_bound = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_periods,
                                                    model.s_landuses, model.s_lmus, model.s_season_types, rule=k_bound, doc='bound on landuse area')

        ##crop bound - this can either be entered as a % of farm area or ha
        ###build bound if turned on
        if crop_area_bound_inc:
            ###setbound using % of farm area
            len_q = sinp.structuralsa['i_len_q']  # number of years in MP model
            keys_q = np.array(['q%s' % i for i in range(len_q)])
            crop_area_percent_qk1 = fun.f_sa(np.array([99999]), sen.sav['bnd_crop_area_percent_qk1'][0:len_q, pinp.crop_landuse_mask_k1], 5)  # 99999 is arbitrary default value which mean skip constraint
            crop_area_bound_qk1 = np.full_like(crop_area_percent_qk1,99999)
            crop_area_bound_qk1[crop_area_percent_qk1!=99999] = (crop_area_percent_qk1 * sum(model.p_area[l] for l in model.s_lmus))[crop_area_percent_qk1!=99999]
            ###setbound using ha of farm area
            crop_area_bound_qk1 = fun.f_sa(crop_area_bound_qk1, sen.sav['bnd_crop_area_qk1'][0:len_q, pinp.crop_landuse_mask_k1], 5)
            crop_area_bound_qk1 = fun.f1_make_pyomo_dict(crop_area_bound_qk1, [keys_q, model.s_crops])
            model.p_crop_area_bound_qk1 = pe.Param(model.s_sequence_year, model.s_crops, default=0, initialize=crop_area_bound_qk1)

            ###constraint
            l_p7 = list(model.s_season_periods)
            def k1_bound(model, q, s, p7, k1, z):
                if p7 == l_p7[-1] and model.p_crop_area_bound_qk1[q,k1]!=99999 and pe.value(model.p_wyear_inc_qs[q, s]):  #bound will not be built if param == 99999
                    return(
                           sum(model.v_phase_area[q,s,p7,z,r,l] * model.p_landuse_area[r, k1] for r in model.s_phases for l in model.s_lmus)
                           == model.p_crop_area_bound_qk1[q,k1])
                else:
                    return pe.Constraint.Skip
            model.con_crop_area_bound = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_periods, model.s_crops, model.s_season_types, rule=k1_bound, doc='bound on total pasture area')

        ##total legume crop area bound
        ###build bound if turned on
        if legume_area_bound_inc:
            total_legume_area_percent = sen.sav['bnd_total_legume_area_percent']
            landuse_is_legume = dict(zip(model.s_crops, np.array([x in sinp.landuse['P'] for x in sinp.general['i_idx_k1']], dtype=int)))
            ###constraint
            l_p7 = list(model.s_season_periods)
            def legume_bound(model, q, s, p7, z):
                if p7 == l_p7[-1] and pe.value(model.p_wyear_inc_qs[q, s]):
                    return(
                           sum(model.v_phase_area[q,s,p7,z,r,l] * model.p_landuse_area[r, k1] * landuse_is_legume[k1] for r in model.s_phases for l in model.s_lmus for k1 in model.s_crops)
                           == sum(model.p_area[l] for l in model.s_lmus) * total_legume_area_percent)
                else:
                    return pe.Constraint.Skip
            model.con_legume_area_bound = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_periods, model.s_season_types, rule=legume_bound, doc='bound on total pasture area')

        ##total pasture area - hence also total crop area
        ###build bound if turned on
        if total_pasture_bound_inc:
            ###setbound  - 99999 is arbitrary default value which mean skip constraint
            len_q = sinp.structuralsa['i_len_q']  # number of years in MP model
            keys_q = np.array(['q%s' % i for i in range(len_q)])
            total_pas_area_percent_q = fun.f_sa(np.array([99999]), sen.sav['bnd_total_pas_area_percent_q'][0:len_q], 5)
            total_pas_area_percent_q = dict(zip(keys_q, total_pas_area_percent_q))
            ###constraint
            l_p7 = list(model.s_season_periods)
            def total_pas_bound(model, q, s, p7, z):
                if p7 == l_p7[-1] and total_pas_area_percent_q[q] != 99999 and pe.value(model.p_wyear_inc_qs[q, s]):
                    return (sum(model.v_phase_area[q,s,p7,z,r,l] * model.p_pasture_area[r,t]
                                for r in model.s_phases for l in model.s_lmus for t in model.s_pastures)
                            == sum(model.p_area[l] for l in model.s_lmus) * total_pas_area_percent_q[q])
                else:
                    return pe.Constraint.Skip
            model.con_total_pas_bound = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_periods, model.s_season_types, rule=total_pas_bound,doc='bound on total pasture area')

        ##bnd area of each pasture
        if pasture_bound_inc:
            ###setbound  - 99999 is arbitrary default value which mean skip constraint
            pas_area_percent_t = fun.f_sa(np.array([99999]), sen.sav['bnd_pas_area_percent_t'][pinp.general['pas_inc_t']], 5)
            keys_t = sinp.general['pastures'][pinp.general['pas_inc_t']]
            pas_area_percent_t = dict(zip(keys_t, pas_area_percent_t))
            ###constraint
            l_p7 = list(model.s_season_periods)
            def pas_bound(model, q, s, p7, z, t):
                if p7 == l_p7[-1] and pas_area_percent_t[t] != 99999 and pe.value(model.p_wyear_inc_qs[q, s]):
                    return (sum(model.v_phase_area[q,s,p7,z,r,l] * model.p_pasture_area[r,t]
                                for r in model.s_phases for l in model.s_lmus)
                            == sum(model.p_area[l] for l in model.s_lmus) * pas_area_percent_t[t])
                else:
                    return pe.Constraint.Skip
            model.con_pas_bound = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_periods,
                                                model.s_season_types, model.s_pastures, rule=pas_bound,doc='bound on each pasture area')

        ##pasture area by lmu
        ###build bound if turned on
        if pasture_lmu_bound_inc:
            ###setbound
            pas_area_l = fun.f_sa(np.array([99999]), sen.sav['bnd_pas_area_l'], 5)  # 99999 is arbitrary default value which mean skip constraint
            pas_area_l = dict(zip(model.s_lmus, pas_area_l))
            ###constraint
            l_p7 = list(model.s_season_periods)
            def pas_bound(model, q, s, p7, z, l):
                if p7 == l_p7[-1] and pe.value(model.p_wyear_inc_qs[q, s]) and pas_area_l[l] != 99999:
                    return (sum(model.v_phase_area[q,s,p7,z,r,l] * model.p_pasture_area[r,t]
                                for r in model.s_phases for t in model.s_pastures)
                            == pas_area_l[l])
                else:
                    return pe.Constraint.Skip
            model.con_pas_bound = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_periods, model.s_season_types, model.s_lmus, rule=pas_bound,doc='bound pasture area by lmu')


        ##bound to limit the number of years of cont lucerne (Nov24 inputs are based on 5yrs of lucerne)
        if cont_phase_bound_inc:
            ###propn of each landuse that can be in a cont phase
            max_yr_cont_k = fun.f_sa(np.array([99999]), sen.sav['max_yr_cont_k'][pinp.all_landuse_mask_k], 5)  # 99999 is arbitrary default value which mean skip constraint
            p_cont_freq = sinp.general['phase_len']/max_yr_cont_k
            p_cont_freq_k = dict(zip(model.s_landuses, p_cont_freq))
            ###constraint
            l_p7 = list(model.s_season_periods)
            def cont_phase_bound(model, q, s, p7, k, l, z):
                if p7 == l_p7[-1] and p_cont_freq_k[k]>0.001 and pe.value(model.p_wyear_inc_qs[q, s]):
                    return(sum(model.v_phase_area[q,s,p7,z,r,l] * model.p_phase_continuous_r[r] * model.p_landuse_area[r, k] for r in model.s_phases)
                           == p_cont_freq_k[k] * sum(model.v_phase_area[q,s,p7,z,r,l] * model.p_landuse_area[r, k] for r in model.s_phases))
                else:
                    return pe.Constraint.Skip
            model.con_cont_phase_bound = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_periods,
                                                    model.s_landuses, model.s_lmus, model.s_season_types, rule=cont_phase_bound, doc='bound on number of years continuous phase can exist for')

        ##biomass grazed bound - proportion of the biomass that is produced that is grazed
        ###This could be expanded to handle a sav with a s2 axis. Currently, it only controls the proportion grazed (s2[1])
        ###build bound if turned on
        if biomass_graze_bound_inc:
            ###setbound using ha of farm area
            biomass_graze_bound_k1 = fun.f_sa(np.array([99999]), sen.sav['bnd_biomass_graze_k1'][pinp.crop_landuse_mask_k1], 5)  # 99999 is arbitrary default value which mean skip constraint
            biomass_graze_bound_k1 = dict(zip(model.s_crops, biomass_graze_bound_k1))

            def k1_graze_bound(model, q, s, k1, z):
                if biomass_graze_bound_k1[k1]!=99999 and pe.value(model.p_wyear_inc_qs[q, s]):  #bound will not be built if param == 99999
                    return (
                        sum(model.v_use_biomass[q,s,p7,z,k1,l,'Graz'] for l in model.s_lmus for p7 in model.s_season_periods)
                        == biomass_graze_bound_k1[k1] * sum(model.v_use_biomass[q,s,p7,z,k1,l,s2] for s2 in model.s_biomass_uses
                                                            for l in model.s_lmus for p7 in model.s_season_periods))
                else:
                    return pe.Constraint.Skip
            model.con_biomass_graze_bound = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_crops, model.s_season_types, rule=k1_graze_bound, doc='bound on biomass grazing')


        if emissions_bnd_inc:
            def emissions(model,q,s):
                '''
                Constrains total farm co2e emissions.

                p7 axis required due to season clustering.

                Note this constraint could be modified to constrain emission intensity or to constrain emissions from each enterprise.
                '''
                #TODO - need to add trees and carbon sold if carbon is sold it doesnt reduce farm footprint.
                if model.p_co2e_limit>0:
                    return sum((sum((suppy.f_sup_emissions(model,q,s,p6,z) + cgzpy.f_grazecrop_emissions(model,q,s,p6,z)
                                     + stubpy.f_cropresidue_consumption_emissions(model,q,s,p6,z) + slppy.f_saltbush_emissions(model,q,s,z,p6)
                                     + paspy.f_pas_emissions(model,q,s,p6,z))*model.p_a_p6_p7[p7,p6,z] for p6 in model.s_feed_periods)
                                + stkpy.f_stock_emissions(model,q,s,p7,z) + stubpy.f_cropresidue_production_emissions(model,q,s,p7,z)
                                + phspy.f_rot_emissions(model, q, s, p7, z) + macpy.f_seeding_harv_fuel_emissions(model, q, s, p7, z)) * model.p_season_seq_prob_qszp7[q,s,z,p7]
                                for z in model.s_season_types for p7 in model.s_season_periods if pe.value(model.p_season_seq_prob_qszp7[q,s,z,p7]) != 0) <= uinp.emissions['co2e_limit']
                else:
                    return pe.Constraint.Skip
            model.con_emissions = pe.Constraint(rule=emissions,doc='co2e emissions')



