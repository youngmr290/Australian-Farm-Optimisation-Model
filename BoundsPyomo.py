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
from CreateModel import model
import pyomo.environ as pe


'''
Bounds

note:
-forcing sale/retention of drys is done in the stock module (there are inputs which user can control this with)
-wether sale age bound is in stock module (controlled via SAV)
'''



def boundarypyomo_local(params):

    ##set bounds to include
    bounds_inc = True #controls all bounds (typically on)
    rot_lobound_inc = False #controls rot bound
    dams_lobound_inc = True #lower bound dams
    dams_upperbound_inc = fun.f_sa(False, sen.sav['bnd_upper_dam_inc'], 5) #upper bound on dams
    total_dams_scanned_bound_inc = np.any(sen.sav['bnd_total_dams_scanned'] != '-') #equal to bound on the total number of scanned dams
    force_5yo_retention_inc = np.any(sen.sav['bnd_propn_dam5_retained'] != '-') #force a propn of 5yo dams to be retained.
    bnd_propn_dams_mated = np.any(sen.sav['bnd_propn_dams_mated_og1'] != '-')
    bnd_sale_twice_drys_inc = fun.f_sa(False, sen.sav['bnd_sale_twice_dry_inc'], 5) #proportion of drys sold (can be sold at either sale opp)
    sr_bound_inc = fun.f_sa(False, sen.sav['bnd_sr_inc'], 5) #controls sr bound
    total_pasture_bound_inc = fun.f_sa(False, sen.sav['bnd_pasarea_inc'], 5)  #bound on total pasture (hence also total crop)
    landuse_bound_inc = False #bound on area of each landuse


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
        try:
            model.del_component(model.p_mask_dams)
            model.del_component(model.p_mask_dams_index)
        except AttributeError:
            pass
        model.p_mask_dams = pe.Param(model.s_k2_birth_dams, model.s_sale_dams, model.s_dvp_dams, model.s_lw_dams,
                                           initialize=params['stock']['p_mask_dams'])

        ##rotations
        ###delete the bound (outside if statement incase the bound was active for last trial)
        try:
            model.del_component(model.con_rotation_lobound)
            model.del_component(model.con_rotation_lobound_index)
        except AttributeError:
            pass
        ###build bound if turned on
        if rot_lobound_inc:
            ###keys to build arrays
            arrays = [model.s_phases, model.s_lmus]
            index_rl = fun.cartesian_product_simple_transpose(arrays)
            ###build array
            rot_lobound_rl = np.zeros((len(model.s_phases), len(model.s_lmus)))
            ###set the bound
            rot_lobound_rl[0,2] = 150
            ###ravel and zip bound and dict
            rot_lobound = rot_lobound_rl.ravel()
            tup_rl = tuple(map(tuple, index_rl))
            rot_lobound = dict(zip(tup_rl, rot_lobound))
            ###constraint
            def rot_lo_bound(model, r, l):
                return model.v_phase_area[r, l] >= rot_lobound[r,l]
            model.con_rotation_lobound = pe.Constraint(model.s_phases, model.s_lmus, rule=rot_lo_bound,
                                                    doc='lo bound for the number of each phase')

        ##total dam min bound - total number includes each dvp (the sheep in a given yr equal total for all dvp divided by the number of dvps in 1 yr)
        ###delete the bound (outside if statement incase the bound was active for last trial)
        try:
            model.del_component(model.con_dam_lobound)
            model.del_component(model.con_dam_lobound_index)
        except AttributeError:
            pass
        ###build bound if turned on
        if dams_lobound_inc:
            ###keys to build arrays for the specified slices
            arrays = [model.s_dvp_dams, model.s_groups_dams]   #more sets can be added here to customise the bound
            index_vg = fun.cartesian_product_simple_transpose(arrays)
            ###build array for the axes of the specified slices
            dams_lowbound_vg = np.zeros((len(model.s_dvp_dams), len(model.s_groups_dams)))
            ###set the bound
            dams_lowbound_vg[4:14,-1] = 2000  #min of 50 bbt
            ###ravel and zip bound and dict
            dams_lowbound = dams_lowbound_vg.ravel()
            tup_tv = tuple(map(tuple, index_vg))
            dams_lowbound = dict(zip(tup_tv, dams_lowbound))

            ###constraint
            def dam_lo_bound(model,v,g1):
                if all(model.p_mask_dams[k2,t,v,w8] == 0
                       for k2 in model.s_k2_birth_dams for t in model.s_sale_dams for w8 in model.s_lw_dams):
                    return pe.Constraint.Skip
                else:
                    return sum(model.v_dams[k2,t,v,a,n,w8,i,y,g1] for k2 in model.s_k2_birth_dams for t in model.s_sale_dams
                               for a in model.s_wean_times for n in model.s_nut_dams for w8 in model.s_lw_dams
                               for i in model.s_tol for y in model.s_gen_merit_dams
                               if model.p_mask_dams[k2,t,v,w8] == 1) \
                           >= dams_lowbound[v,g1]
            model.con_dam_lobound = pe.Constraint(model.s_dvp_dams, model.s_groups_dams, rule=dam_lo_bound,
                                                    doc='min number of all dams')

        ##dams upper bound - specified by k2 & v and totalled across other axes
        ###delete the bound (outside if statement incase the bound was active for last trial)
        try:
            model.del_component(model.con_dam_upperbound)
            model.del_component(model.con_dam_upperbound_index)
        except AttributeError:
            pass
        ###build bound if turned on
        if dams_upperbound_inc:
            ###keys to build arrays for the specified slices
            arrays = [model.s_sale_dams, model.s_dvp_dams]   #more sets can be added here to customise the bound
            index_tv = fun.cartesian_product_simple_transpose(arrays)
            ###build array for the axes of the specified slices
            dams_upperbound_tv = np.full((len(model.s_sale_dams), len(model.s_dvp_dams)), np.inf)
            ###set the bound
            dams_upperbound_tv[0:2, 0:14] = 0  #no dam sales before dvp14
            dams_upperbound_tv[0:1, 3:4] = np.inf   #allow sale after shearing t[0] for dams dvp3
            ###ravel and zip bound and dict
            dams_upperbound = dams_upperbound_tv.ravel()
            tup_tv = tuple(map(tuple, index_tv))
            dams_upperbound = dict(zip(tup_tv, dams_upperbound))
            ###constraint
            def f_dam_upperbound(model, t, v):
                if dams_upperbound[t, v]==np.inf or all(model.p_mask_dams[k2,t,v,w8] == 0
                       for k2 in model.s_k2_birth_dams for t in model.s_sale_dams for w8 in model.s_lw_dams):
                    return pe.Constraint.Skip
                else:
                    return sum(model.v_dams[k28,t,v,a,n,w8,i,y,g1] for k28 in model.s_k2_birth_dams
                               for a in model.s_wean_times for n in model.s_nut_dams for w8 in model.s_lw_dams
                               for i in model.s_tol for y in model.s_gen_merit_dams for g1 in model.s_groups_dams
                               if model.p_mask_dams[k28,t,v,w8] == 1 #if removes the masked out dams so they dont show up in .lp output.
                               ) <= dams_upperbound[t, v]
            model.con_dam_upperbound = pe.Constraint(model.s_sale_dams, model.s_dvp_dams, rule=f_dam_upperbound,
                                                    doc='max number of dams_tv')

        ##total dams scanned. Sums dams in all scanning dvps
        ###delete the bound (outside if statement incase the bound was active for last trial)
        try:
            model.del_component(model.con_total_dams_scanned)
        except AttributeError:
            pass
        ###build bound if turned on
        if total_dams_scanned_bound_inc:
            ###set the bound
            total_dams_scanned = fun.f_sa(np.inf, sen.sav['bnd_total_dams_scanned'], 5)
            ###scan dvps
            scan_v = list(params['stock']['p_scan_v_dams'])
            ###constraint - sum all mated dams in scan dvp.
            def f_total_dams_scanned(model):
                if propn_dams_retained == np.inf:
                    pe.Constraint.Skip
                else:
                    return sum(model.v_dams[k28,t,v,a,n,w8,i,y,g1] for k28 in model.s_k2_birth_dams for t in model.s_sale_dams
                               for v in model.s_dvp_dams for a in model.s_wean_times for n in model.s_nut_dams for w8 in model.s_lw_dams
                               for i in model.s_tol for y in model.s_gen_merit_dams for g1 in model.s_groups_dams
                               if model.p_mask_dams[k28,t,v,w8] == 1 and v in scan_v and k28 != 'NM-0') \
                       == total_dams_scanned
            model.con_total_dams_scanned = pe.Constraint(rule=f_total_dams_scanned, doc='total dams scanned')

        ##force 5yo dam retention - fix a proportion of dams at 6yo scanning dvp.
        ###delete the bound (outside if statement incase the bound was active for last trial)
        try:
            model.del_component(model.con_retention_5yo_dams)
        except AttributeError:
            pass
        ###build bound if turned on
        if force_5yo_retention_inc:
            ###set the bound
            propn_dams_retained = fun.f_sa(np.inf, sen.sav['bnd_propn_dam5_retained'], 5)
            ###5yr scan dvp
            scan5_v = list(params['stock']['p_scan_v_dams'])[4]
            ###6yr scan dvp
            scan6_v = list(params['stock']['p_scan_v_dams'])[5]
            ###constraint - sum all mated dams in scan dvp.
            def retention_5yo_dams(model):
                if propn_dams_retained == np.inf:
                    pe.Constraint.Skip
                else:
                    return (propn_dams_retained) * sum(model.v_dams[k28,t,v,a,n,w8,i,y,g1] for k28 in model.s_k2_birth_dams
                                                       for t in model.s_sale_dams for v in model.s_dvp_dams
                                                       for a in model.s_wean_times for n in model.s_nut_dams for w8 in model.s_lw_dams
                                                       for i in model.s_tol for y in model.s_gen_merit_dams for g1 in model.s_groups_dams
                                                       if model.p_mask_dams[k28,t,v,w8] == 1 and v in scan5_v) \
                       == sum(model.v_dams[k28,t,v,a,n,w8,i,y,g1] for k28 in model.s_k2_birth_dams
                              for t in model.s_sale_dams for v in model.s_dvp_dams for a in model.s_wean_times
                              for n in model.s_nut_dams for w8 in model.s_lw_dams for i in model.s_tol
                              for y in model.s_gen_merit_dams for g1 in model.s_groups_dams
                              if model.p_mask_dams[k28,t,v,w8] == 1 and v in scan6_v)
            model.con_retention_5yo_dams = pe.Constraint(rule=retention_5yo_dams, doc='force retention of 5yo dams')

        ##bound to fix the proportion of dams being mated - typically used to exclude yearlings
        ###delete the bound (outside if statement incase the bound was active for last trial)
        try:
            model.del_component(model.con_propn_dams_mated_index)
            model.del_component(model.con_propn_dams_mated)
        except AttributeError:
            pass
        ###build bound if turned on
        if bnd_propn_dams_mated:
            ###build param - inf values are skipped in the constraint building so inf means the model can optimise the propn mated
            try:
                model.del_component(model.p_prop_dams_mated_index)
                model.del_component(model.p_prop_dams_mated)
            except AttributeError:
                pass
            model.p_prop_dams_mated = pe.Param(model.s_dvp_dams, model.s_groups_dams, initialize=params['stock']['p_prop_dams_mated'])
            ###constraint
            def f_propn_dams_mated(model, v, g1):
                if model.p_prop_dams_mated[v, g1]==np.inf or all(model.p_mask_dams[k2,t, v, w8] == 0
                                      for k2 in model.s_k2_birth_dams for t in model.s_sale_dams for w8 in model.s_lw_dams):
                    return pe.Constraint.Skip
                else:
                    return sum(model.v_dams['NM-0',t,v,a,n,w8,i,y,g1] for t in model.s_sale_dams
                               for a in model.s_wean_times for n in model.s_nut_dams for w8 in model.s_lw_dams
                               for i in model.s_tol for y in model.s_gen_merit_dams
                               if model.p_mask_dams['NM-0',t,v,w8] == 1
                               ) == sum(model.v_dams[k2,t,v,a,n,w8,i,y,g1] for k2 in model.s_k2_birth_dams for t in model.s_sale_dams
                               for a in model.s_wean_times for n in model.s_nut_dams for w8 in model.s_lw_dams
                               for i in model.s_tol for y in model.s_gen_merit_dams
                               if model.p_mask_dams[k2,t,v,w8] == 1
                                        ) * (1 - model.p_prop_dams_mated[v, g1])
            model.con_propn_dams_mated = pe.Constraint(model.s_dvp_dams, model.s_groups_dams, rule=f_propn_dams_mated,
                                                       doc='proportion of dams mated')

        ##bound to fix the proportion of twice dry dams sold. Proportion of twice dry dams is an input in uinp ce[2, ....].
        ###delete the bound (outside if statement incase the bound was active for last trial)
        try:
            model.del_component(model.con_propn_drys_sold_index)
            model.del_component(model.con_propn_drys_sold)
        except AttributeError:
            pass
        ###build bound if turned on
        if bnd_sale_twice_drys_inc:
            '''Constraint forces x percent of the dry dams to be sold (all the twice drys). They can be sold in either sale op (after scanning or at shearing).
             Thus the drys just have to be sold sometime between scanning and the following prejoining.'''
            ###build param - inf values are skipped in the constraint building so inf means the model can optimise the propn mated
            try:
                model.del_component(model.p_prop_twice_dry_dams_index)
                model.del_component(model.p_prop_twice_dry_dams)
            except AttributeError:
                pass
            model.p_prop_twice_dry_dams = pe.Param(model.s_dvp_dams, model.s_tol, model.s_gen_merit_dams,
                                                   model.s_groups_dams, initialize=params['stock']['p_prop_twice_dry_dams'])

            l_v1 = list(model.s_dvp_dams)
            scan_v = list(params['stock']['p_scan_v_dams'])
            prejoin_v = list(params['stock']['p_prejoin_v_dams'])[1:] #remove the start dvp which is not a real prejoin dvp (it just has type condense).
            next_prejoin_v = prejoin_v[1:] #dvp before following prejoining

            ###constraint
            def f_propn_drys_sold(model, v, i, y, g1):
                if v in scan_v[:-1] and model.p_prop_twice_dry_dams[v, i, y, g1]!=0: #use 00 numbers at scanning. Dont want to include the last prejoining dvp because there is no sale limit in the last year.
                    idx_scan = scan_v.index(v) #which prejoining is the current v
                    idx_v_next_prejoin = next_prejoin_v[idx_scan] #all the twice drys must be sold by the following prejoining
                    v_sale = l_v1[l_v1.index(idx_v_next_prejoin) - 1]
                    return sum(model.v_dams['00-0','t2',v_sale,a,n,w8,i,y,g1]
                               for a in model.s_wean_times for n in model.s_nut_dams for w8 in model.s_lw_dams
                               ) == sum(model.v_dams['00-0',t,v,a,n,w8,i,y,g1] for t in model.s_sale_dams
                               for a in model.s_wean_times for n in model.s_nut_dams for w8 in model.s_lw_dams
                                        if model.p_mask_dams['00-0',t,v,w8] == 1
                                        ) * model.p_prop_twice_dry_dams[v, i, y, g1]
                else:
                    return pe.Constraint.Skip
            model.con_propn_drys_sold = pe.Constraint(model.s_dvp_dams, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, rule=f_propn_drys_sold,
                                                       doc='proportion of dry dams sold each year')

        ##SR - this can't set the sr on an actual pasture but it means different pastures provide a different level of carry capacity although nothing fixes sheep to that pasture
        ###delete the bound (outside if statement incase the bound was active for last trial)
        try:
            model.del_component(model.con_SR_bound)
            model.del_component(model.con_SR_bound_index)
        except AttributeError:
            pass
        ###build bound
        if sr_bound_inc:
            ###initilise
            pasture_dse_carry = {} #populate straight into dict
            ### set bound - carry cap of each ha of each pasture
            for t, pasture in enumerate(sinp.general['pastures'][pinp.general['pas_inc']]):
                pasture_dse_carry[pasture] = pinp.sheep['i_sr_constraint_t'][t]
            ###constraint
            def SR_bound(model, p6):
                return(
                - sum(model.v_phase_area[r, l] * model.p_pasture_area[r, t] * pasture_dse_carry[t] for r in model.s_phases for l in model.s_lmus for t in model.s_pastures)
                + sum(model.v_sire[g0] * model.p_dse_sire[p6,g0] for g0 in model.s_groups_sire if model.p_dse_sire[p6,g0]!=0)
                + sum(sum(model.v_dams[k2,t1,v1,a,n1,w1,z,i,y1,g1] * model.p_dse_dams[k2,p6,t1,v1,a,n1,w1,z,i,y1,g1]
                         for k2 in model.s_k2_birth_dams for t1 in model.s_sale_dams for v1 in model.s_dvp_dams for n1 in model.s_nut_dams
                         for w1 in model.s_lw_dams for y1 in model.s_gen_merit_dams for g1 in model.s_groups_dams if model.p_dse_dams[k2,p6,t1,v1,a,n1,w1,z,i,y1,g1]!=0)
                    + sum(model.v_offs[k3,k5,t3,v3,n3,w3,z,i,a,x,y3,g3]  * model.p_dse_offs[k3,k5,p6,t3,v3,n3,w3,z,i,a,x,y3,g3]
                          for k3 in model.s_k3_damage_offs for k5 in model.s_k5_birth_offs for t3 in model.s_sale_offs for v3 in model.s_dvp_offs
                          for n3 in model.s_nut_offs for w3 in model.s_lw_offs for x in model.s_gender for y3 in model.s_gen_merit_offs for g3 in model.s_groups_offs if model.p_dse_offs[k3,k5,p6,t3,v3,n3,w3,z,i,a,x,y3,g3]!=0)
               for a in model.s_wean_times for z in model.s_season_types for i in model.s_tol) ==0)
            model.con_SR_bound = pe.Constraint(model.s_feed_periods, rule=SR_bound,
                                                doc='stocking rate bound for each feed period')


        ##landuse bound
        ###delete the bound (outside if statement incase the bound was active for last trial)
        try:
            model.del_component(model.con_landuse_bound)
        except AttributeError:
            pass
        ###build bound if turned on
        if landuse_bound_inc:
            ##initilise bound - note zero is the equivalent of no bound
            landuse_bound_k = pd.Series(0,index=model.s_landuses) #use landuse2 because that is the expanded version of pasture phases eg t, tr not just tedera
            ##set bound - note that setting to zero is the equivalent of no bound
            landuse_bound_k.iloc[0] = 50
            ###dict
            landuse_area_bound = dict(landuse_bound_k)
            ###constraint
            def k_bound(model, k):
                if landuse_area_bound[k]!=0:  #bound will not be built if param == 0
                    return(
                           sum(model.v_phase_area[r, l] * model.p_landuse_area[r, k] for r in model.s_phases for l in model.s_lmus for t in model.s_pastures)
                           == landuse_area_bound[k])
                else:
                    pe.Constraint.Skip
            model.con_pas_bound = pe.Constraint(model.s_landuses, rule=k_bound, doc='bound on total pasture area')

        ##total pasture area - hence also total crop area
        ###delete the bound (outside if statement incase the bound was active for last trial)
        try:
            model.del_component(model.con_landuse_bound)
        except AttributeError:
            pass
        ###build bound if turned on
        if total_pasture_bound_inc:
            ###setbound
            total_pas_area = sen.sav['bnd_total_pas_area']
            ###constraint
            try:
                model.del_component(model.con_pas_bound)
            except AttributeError:
                pass
            def pas_bound(model):
                return (
                        sum(model.v_phase_area[r,l] * model.p_pasture_area[r,t] for r in model.s_phases for l in
                            model.s_lmus for t in model.s_pastures)
                        == total_pas_area)
            model.con_pas_bound = pe.Constraint(rule=pas_bound,doc='bound on total pasture area')



 


