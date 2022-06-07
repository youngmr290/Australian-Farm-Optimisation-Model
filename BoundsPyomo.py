'''
THis is where you can set up constraints on individual variables
withing a set of variables. This will act as bounds.

bounds are all controlled from this module
'''

import numpy as np
import pandas as pd
import time

import Functions as fun
import Sensitivity as sen
import UniversalInputs as uinp
import StructuralInputs as sinp
import PropertyInputs as pinp
import pyomo.environ as pe


'''
Bounds

note:
-forcing sale/retention of drys is done in the stock module (there are inputs which user can control this with)
-wether sale age bound is in stock module (controlled via SAV)
'''



def f1_boundarypyomo_local(params, model):

    ##set bounds to include
    bounds_inc = True #controls all bounds (typically on)
    rot_lobound_inc = fun.f_sa(False, sen.sav['bnd_rotn_inc'], 5)  #controls rot bound
    slp_area_inc = np.any(sen.sav['bnd_slp_area_l'] != '-') #control the area of salt land pasture
    sup_lobound_inc = False #controls sup feed bound
    dams_lobound_inc = fun.f_sa(False, sen.sav['bnd_lo_dam_inc'], 5) #lower bound dams
    dams_upbound_inc = fun.f_sa(False, sen.sav['bnd_up_dam_inc'], 5) #upper bound on dams
    offs_lobound_inc = fun.f_sa(False, sen.sav['bnd_lo_off_inc'], 5) #lower bound offs
    offs_upbound_inc = fun.f_sa(False, sen.sav['bnd_up_off_inc'], 5) #upper bound on offs
    total_dams_scanned_bound_inc = np.any(sen.sav['bnd_total_dams_scanned'] != '-') #equal to bound on the total number of mated dams at scanning
    force_5yo_retention_inc = np.any(sen.sav['bnd_propn_dam5_retained'] != '-') #force a propn of 5yo dams to be retained.
    bnd_propn_dams_mated_inc = np.any(sen.sav['bnd_propn_dams_mated_og1'] != '-')
    bnd_sale_twice_drys_inc = fun.f_sa(False, sen.sav['bnd_sale_twice_dry_inc'], 5) #proportion of drys sold (can be sold at either sale opp)
    bnd_dry_retained_inc = fun.f_sa(False, np.any(pinp.sheep['i_dry_retained_forced_o']), 5) #force the retention of drys in t[0] (t[1] is handled in the generator.
    sr_bound_inc = fun.f_sa(False, sen.sav['bnd_sr_inc'], 5) #controls sr bound
    total_pasture_bound_inc = fun.f_sa(False, sen.sav['bnd_pasarea_inc'], 5)  #bound on total pasture (hence also total crop)
    landuse_bound_inc = False #bound on area of each landuse (which is the sum of all the phases for that landuse)


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
        model.p_mask_dams = pe.Param(model.s_k2_birth_dams, model.s_sale_dams, model.s_dvp_dams, model.s_lw_dams, model.s_season_types,
                                     model.s_groups_dams, default=0, initialize=params['stock']['p_mask_dams'])

        ##rotations
        ###build bound if turned on
        if rot_lobound_inc:
            ###keys to build arrays
            arrays = [model.s_phases, model.s_lmus]
            index_rl = fun.cartesian_product_simple_transpose(arrays)
            ###build array
            #rot_lobound_rl = np.zeros((len(model.s_phases), len(model.s_lmus)))
            ###set the bound
            rot_lobound_rl = fun.f_sa(np.array([0],dtype=float), sen.sav['rot_lobound_rl'], 5)
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


        ##salt land pasture area
        if slp_area_inc:
            ###set the bound
            slp_area_bnd_l = fun.f_sa(np.array([999999],dtype=float), sen.sav['bnd_slp_area_l'], 5) #999999 is arbitrary default value which mean skip constraint
            ###ravel and zip bound and dict
            slp_area = dict(zip(model.s_lmus, slp_area_bnd_l))
            ###constraint
            l_p7 = list(model.s_season_periods)
            def slp_area_bound(model, q, s, z, l):
                if pe.value(model.p_wyear_inc_qs[q, s]) and slp_area_bnd_l[l] != 999999:
                    return model.v_slp_ha[q,s,z,l] == slp_area[l]
                else:
                    return pe.Constraint.Skip
            model.con_rotation_lobound = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_types, model.s_lmus, rule=slp_area_bound,
                                                    doc='bound for the area of salt land pasture on each lmu')


        ##bound on livestock supplementary feed.
        if sup_lobound_inc:
            def sup_upper_bound(model, q, s, z):
                if pe.value(model.p_wyear_inc_qs[q, s]):
                    return sum(model.v_sup_con[q,s,z,k,g,f,p6] for k in model.s_crops for g in model.s_grain_pools for f in model.s_feed_pools
                    for p6 in model.s_feed_periods) >= 115
            model.con_sup_upper_bound = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_types, rule=sup_upper_bound, doc='upper bound for livestock sup feed')


        ##dam lo bound. (the sheep in a given yr equal total for all dvp divided by the number of dvps in 1 yr)
        if dams_lobound_inc:
            '''
            Lower bound dams.
            
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
            def f_dam_lobound(model, q, s, t, v, z, g1):
                if (pe.value(model.p_wyear_inc_qs[q, s]) and model.p_dams_lobound[t,v,z,g1]!=0
                    and any(model.p_mask_dams[k2,t,v,w8,z,g1] != 0 for k2 in model.s_k2_birth_dams for w8 in model.s_lw_dams)):
                    return sum(model.v_dams[q,s,k2,t,v,a,n,w8,z,i,y,g1] for k2 in model.s_k2_birth_dams
                               for a in model.s_wean_times for n in model.s_nut_dams for w8 in model.s_lw_dams
                               for i in model.s_tol for y in model.s_gen_merit_dams
                               if pe.value(model.p_mask_dams[k2,t,v,w8,z,g1]) == 1
                               ) >= model.p_dams_lobound[t,v,z,g1]
                else:
                    return pe.Constraint.Skip
            model.con_dams_lobound = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_sale_dams
                                                   , model.s_dvp_dams, model.s_season_types, model.s_groups_dams
                                                   , rule=f_dam_lobound, doc='min number of dams')

        ##dams upper bound
        if dams_upbound_inc:
            '''
            Upper bound dams.

            The constraint sets can be changed for different analysis (this is preferred rather than creating 
            a new lobound because that keeps this module smaller and easier to navigate etc).
            Typically set using SAV however for quick and dirty debugging the code below can be uncommented.

            Process to add/remove constraints sets:

                a) add/remove the set from the constraint below (will also need to add/remove from the sum)
                b) update empty array initialisation in sensitivity.py
                c) update param creation in sgen (param index will need the set added/removed)
            '''
            ##set bound using SAV
            model.p_dams_upbound = pe.Param(model.s_sale_dams, model.s_dvp_dams, model.s_season_types, model.s_groups_dams,
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
            def f_dam_upbound(model, q, s, t, v, z, g1):
                if (pe.value(model.p_wyear_inc_qs[q, s]) and model.p_dams_upbound[t,v,z,g1]!=np.inf
                    and any(model.p_mask_dams[k2,t,v,w8,z,g1] != 0 for k2 in model.s_k2_birth_dams for w8 in model.s_lw_dams)):
                    return sum(model.v_dams[q,s,k28,t,v,a,n,w8,z,i,y,g1] for k28 in model.s_k2_birth_dams
                               for a in model.s_wean_times for n in model.s_nut_dams for w8 in model.s_lw_dams
                               for i in model.s_tol for y in model.s_gen_merit_dams
                               if pe.value(model.p_mask_dams[k28,t,v,w8,z,g1]) == 1 #if removes the masked out dams so they don't show up in .lp output.
                               ) <= model.p_dams_upbound[t,v,z,g1]
                else:
                    return pe.Constraint.Skip
            model.con_dams_upbound = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_sale_dams
                                                   , model.s_dvp_dams, model.s_season_types, model.s_groups_dams
                                                   , rule=f_dam_upbound, doc='max number of dams_tv')

        ##offs lo bound
        if offs_lobound_inc:
            '''
            Lower bound offs.

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
            def f_off_lobound(model, q, s, k3, t, v, z, x, g3):
                if pe.value(model.p_wyear_inc_qs[q, s]):
                    return sum(model.v_offs[q,s,k3,k5,t,v,n3,w8,z,i,a,x,y3,g3]
                               for k5 in model.s_k5_birth_offs for a in model.s_wean_times for n3 in model.s_nut_offs
                               for w8 in model.s_lw_offs for i in model.s_tol for y3 in model.s_gen_merit_offs
                               ) >= model.p_offs_lobound[k3,t,v,z,x,g3]
                else:
                    return pe.Constraint.Skip
            model.con_offs_lobound = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_k3_damage_offs, model.s_sale_offs
                                                   , model.s_dvp_offs, model.s_season_types, model.s_gender, model.s_groups_offs
                                                   , rule=f_off_lobound, doc='min number of offs')

        ##offs upper bound
        if offs_upbound_inc:
            '''
            Upper bound offs.

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
                if pe.value(model.p_wyear_inc_qs[q, s]):
                    return sum(model.v_offs[q,s,k3,k5,t,v,n3,w8,z,i,a,x,y3,g3]
                               for k5 in model.s_k5_birth_offs for a in model.s_wean_times for n3 in model.s_nut_offs
                               for w8 in model.s_lw_offs for i in model.s_tol for y3 in model.s_gen_merit_offs
                               ) <= model.p_offs_upbound[k3,t,v,z,x,g3]
                else:
                    return pe.Constraint.Skip
            model.con_offs_upbound = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_k3_damage_offs, model.s_sale_offs
                                                   , model.s_dvp_offs, model.s_season_types, model.s_gender, model.s_groups_offs
                                                   , rule=f_off_upbound, doc='max number of offs')


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

        ##bound to fix the proportion of dams being mated - typically used to exclude mating yearlings
        #todo this causes sheep to become infeasible in the DSP model. Will need to revisit.
        ###build bound if turned on
        if bnd_propn_dams_mated_inc:
            ###build param - inf values are skipped in the constraint building so inf means the model can optimise the propn mated
            model.p_prop_dams_mated = pe.Param(model.s_dvp_dams, model.s_season_types, model.s_groups_dams, default=0, initialize=params['stock']['p_prop_dams_mated'])
            ###constraint
            #todo add an i axis to the constraint
            def f_propn_dams_mated(model, q, s, v, z, g1):
                if (model.p_prop_dams_mated[v,z,g1]==np.inf or not pe.value(model.p_wyear_inc_qs[q, s]) or
                        all(model.p_mask_dams[k2,t,v,w8,z,g1] == 0 or v=='dv00'   #skip if DVP0 which is a non-mating period in o[0]
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
                               if pe.value(model.p_mask_dams[k2,t,v,w8,z,g1]) == 1
                                        ) * (1 - model.p_prop_dams_mated[v,z,g1])
            model.con_propn_dams_mated = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_dvp_dams, model.s_season_types, model.s_groups_dams, rule=f_propn_dams_mated,
                                                       doc='proportion of dams mated')

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
                                                   model.s_groups_dams, initialize=params['stock']['p_prop_twice_dry_dams'])

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
            model.con_propn_drys_sold = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_dvp_dams, model.s_lw_dams, model.s_season_types, model.s_tol,
                                                      model.s_gen_merit_dams, model.s_groups_dams, rule=f_propn_drys_sold,
                                                       doc='proportion of dry dams sold each year')

        ##bound to force the retention of drys until the dvp when other ewes are sold.
        # The bound is only for t[0] (sale at shearing) t[1] (sale at scanning) is handled in the generator.
        if bnd_dry_retained_inc:
            ###build param
            model.p_prop_dry_t0_dams = pe.Param(model.s_dvp_dams, model.s_wean_times, model.s_nut_dams, model.s_lw_dams, model.s_season_types, model.s_tol,
                                             model.s_gen_merit_dams, model.s_groups_dams, initialize=params['stock']['p_prop_dry_t0_dams'])
            model.p_drys_retained = pe.Param(model.s_dvp_dams, model.s_season_types, model.s_groups_dams,
                                             initialize=params['stock']['p_drys_retained'])

            ###constraint
            def f_retention_drys(model, q, s, v, z, i, g1):
                '''Force the model so that the drys can only be sold when the other ewes are sold (essentially forcing the retention of drys).
                   The number of drys sold must be less than the sum of the other k2 slices'''
                #todo add birth timing to p_prop_dry_t0_dams when gbal is activated
                if pe.value(model.p_wyear_inc_qs[q, s]) and any(model.p_mask_dams['00-0','t0',v,w,z,g1] for w in model.s_lw_dams)!=0 and model.p_drys_retained[v,z,g1]!=0:
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
            pasture_dse_carry = {} #populate straight into dict
            ### set bound - carry cap of each ha of each pasture
            for t, pasture in enumerate(sinp.general['pastures'][pinp.general['pas_inc']]):
                pasture_dse_carry[pasture] = pinp.sheep['i_sr_constraint_t'][t]
            ###param - propn of each fp used in the SR
            ###constraint
            l_p7 = list(model.s_season_periods)
            def SR_bound(model, q, s, p7, z):
                if p7 == l_p7[-1] and pe.value(model.p_wyear_inc_qs[q, s]):
                    rhs_dse = sum(model.v_phase_area[q, s, p7, z, r, l] * model.p_pasture_area[r, t] * pasture_dse_carry[t] for r in model.s_phases for l in model.s_lmus for t in model.s_pastures)
                    dse = sum((sum(model.v_sire[q,s,g0] * model.p_dse_sire[p6,z,g0] for g0 in model.s_groups_sire if pe.value(model.p_dse_sire[p6,z,g0])!=0)
                             + sum(sum(model.v_dams[q,s,k2,t1,v1,a,n1,w1,z,i,y1,g1] * model.p_dse_dams[k2,p6,t1,v1,a,n1,w1,z,i,y1,g1]
                                       for k2 in model.s_k2_birth_dams for t1 in model.s_sale_dams for v1 in model.s_dvp_dams for n1 in model.s_nut_dams
                                       for w1 in model.s_lw_dams for y1 in model.s_gen_merit_dams for g1 in model.s_groups_dams
                                       if pe.value(model.p_dse_dams[k2,p6,t1,v1,a,n1,w1,z,i,y1,g1])!=0)
                                  + sum(model.v_offs[q,s,k3,k5,t3,v3,n3,w3,z,i,a,x,y3,g3] * model.p_dse_offs[k3,k5,p6,t3,v3,n3,w3,z,i,a,x,y3,g3]
                                        for k3 in model.s_k3_damage_offs for k5 in model.s_k5_birth_offs for t3 in model.s_sale_offs for v3 in model.s_dvp_offs
                                        for n3 in model.s_nut_offs for w3 in model.s_lw_offs for x in model.s_gender for y3 in model.s_gen_merit_offs for g3 in model.s_groups_offs
                                        if pe.value(model.p_dse_offs[k3,k5,p6,t3,v3,n3,w3,z,i,a,x,y3,g3])!=0)
                                 for a in model.s_wean_times for z in model.s_season_types for i in model.s_tol))
                            * model.p_wg_propn_p6z[p6,z]
                            for p6 in model.s_feed_periods)
                    return dse == rhs_dse
                else:
                    return pe.Constraint.Skip
            model.con_SR_bound = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_periods, model.s_season_types, rule=SR_bound,
                                                doc='stocking rate bound for each feed period')


        ##landuse bound
        ###build bound if turned on
        if landuse_bound_inc:
            ##initilise bound - note zero is the equivalent of no bound
            landuse_bound_k = pd.Series(0,index=model.s_landuses) #use landuse2 because that is the expanded version of pasture phases e.g. t, tr not just tedera
            ##set bound - note that setting to zero is the equivalent of no bound
            landuse_bound_k.iloc[0] = 50
            ###dict
            landuse_area_bound = dict(landuse_bound_k)
            ###constraint
            l_p7 = list(model.s_season_periods)
            def k_bound(model, q, s, p7, k, z):
                if p7 == l_p7[-1] and landuse_area_bound[k]!=0 and pe.value(model.p_wyear_inc_qs[q, s]):  #bound will not be built if param == 0
                    return(
                           sum(model.v_phase_area[q,s,p7,z,r,l] * model.p_landuse_area[r, k] for r in model.s_phases for l in model.s_lmus)
                           == landuse_area_bound[k])
                else:
                    return pe.Constraint.Skip
            model.con_landuse_bound = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_periods, model.s_landuses, model.s_season_types, rule=k_bound, doc='bound on total pasture area')

        ##total pasture area - hence also total crop area
        ###build bound if turned on
        if total_pasture_bound_inc:
            ###setbound
            total_pas_area = sen.sav['bnd_total_pas_area']
            ###constraint
            l_p7 = list(model.s_season_periods)
            def pas_bound(model, q, s, p7, z):
                if p7 == l_p7[-1] and pe.value(model.p_wyear_inc_qs[q, s]):
                    return (sum(model.v_phase_area[q,s,p7,z,r,l] * model.p_pasture_area[r,t]
                                for r in model.s_phases for l in model.s_lmus for t in model.s_pastures)
                            == total_pas_area)
                else:
                    return pe.Constraint.Skip
            model.con_pas_bound = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_periods, model.s_season_types, rule=pas_bound,doc='bound on total pasture area')



 


