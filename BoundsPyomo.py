'''
THis is where you can set up constraints on individual variables
withing a set of variables. This will act as bounds.

bounds are all controlled from this module
'''

import numpy as np
import pandas as pd

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
    dams_lobound_inc = False #lower bound dams
    dams_upperbound_inc = True #upper bound on dams
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

        ##rotations
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
            try:
                model.del_component(model.con_rotation_lobound)
                model.del_component(model.con_rotation_lobound_index)
            except AttributeError:
                pass
            def rot_lo_bound(model, r, l):
                return model.v_phase_area[r, l] >= rot_lobound[r,l]
            model.con_rotation_lobound = pe.Constraint(model.s_phases, model.s_lmus, rule=rot_lo_bound,
                                                    doc='lo bound for the number of each phase')

        ##total dam min bound - total number includes each dvp (the sheep in a given yr equal total for all dvp divided by the number of dvps in 1 yr)
        ##to customise the bound could make it more like the upper bound below
        if dams_lobound_inc:
            ###set the bound
            dam_lobound = 15000
            ###constraint
            try:
                model.del_component(model.con_dam_lobound)
            except AttributeError:
                pass
            def dam_lo_bound(model):
                return sum(model.v_dams[k28,t,v,a,n,w8,i,y,g1] for k28 in model.s_k2_birth_dams for t in model.s_sale_dams
                           for v in model.s_dvp_dams for a in model.s_wean_times for n in model.s_nut_dams for w8 in model.s_lw_dams
                           for i in model.s_tol for y in model.s_gen_merit_dams for g1 in model.s_groups_dams
                           if any(model.p_numbers_req_dams[k28,k29,t,v,a,n,w8,i,y,g1,g9,w9] == 1 for k29 in model.s_k2_birth_dams
                                  for w9 in model.s_lw_dams for g9 in model.s_groups_dams)) \
                       >= dam_lobound
            model.con_dam_lobound = pe.Constraint(rule=dam_lo_bound,
                                                    doc='min number of all dams')

        ##dams upper bound - specified by k2 & v and totalled across other axes
        #todo need a parameter so that the upperbound can be masked by mask_w8vars to save those variables being displayed in .lp
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
            try:
                model.del_component(model.con_dam_upperbound)
                model.del_component(model.con_dam_upperbound_index)
            except AttributeError:
                pass
            def f_dam_upperbound(model, t, v):
                if dams_upperbound[t, v]==np.inf:
                    return pe.Constraint.Skip
                else:
                    return sum(model.v_dams[k28,t,v,a,n,w8,i,y,g1] for k28 in model.s_k2_birth_dams
                               for a in model.s_wean_times for n in model.s_nut_dams for w8 in model.s_lw_dams
                               for i in model.s_tol for y in model.s_gen_merit_dams for g1 in model.s_groups_dams
                               ) <= dams_upperbound[t, v]
            model.con_dam_upperbound = pe.Constraint(model.s_sale_dams, model.s_dvp_dams, rule=f_dam_upperbound,
                                                    doc='max number of dams_tv')

        ##bound to fix the proportion of dams being mated - typically used to exclude yearlings
        if np.any(sen.sav['bnd_propn_dams_mated_og1']!='-'):
            ###build param - inf values are skipped in the constraint building so inf means the model can optimise the propn mated
            try:
                model.del_component(model.p_prop_dams_mated_index)
                model.del_component(model.p_prop_dams_mated)
            except AttributeError:
                pass
            model.p_prop_dams_mated = pe.Param(model.s_dvp_dams, model.s_groups_dams, initialize=params['stock']['p_prop_dams_mated'])
            ###constraint
            try:
                model.del_component(model.con_propn_dams_mated_index)
                model.del_component(model.con_propn_dams_mated)
            except AttributeError:
                pass
            def f_propn_dams_mated(model, v, g1):
                if model.p_prop_dams_mated[v, g1]==np.inf:
                    return pe.Constraint.Skip
                else:
                    return sum(model.v_dams['NM-0',t,v,a,n,w8,i,y,g1] for t in model.s_sale_dams
                               for a in model.s_wean_times for n in model.s_nut_dams for w8 in model.s_lw_dams
                               for i in model.s_tol for y in model.s_gen_merit_dams
                               ) == sum(model.v_dams[k2,t,v,a,n,w8,i,y,g1] for k2 in model.s_k2_birth_dams for t in model.s_sale_dams
                               for a in model.s_wean_times for n in model.s_nut_dams for w8 in model.s_lw_dams
                               for i in model.s_tol for y in model.s_gen_merit_dams) * (1 - model.p_prop_dams_mated[v, g1])
            model.con_propn_dams_mated = pe.Constraint(model.s_dvp_dams, model.s_groups_dams, rule=f_propn_dams_mated,
                                                       doc='proportion of dams mated')

        ##SR - this can't set the sr on an actual pasture but it means different pastures provide a different level of carry capacity although nothing fixes sheep to that pasture
        if sr_bound_inc:
            ###initilise
            pasture_dse_carry = {} #populate straight into dict
            ### set bound - carry cap of each ha of each pasture
            for t, pasture in enumerate(sinp.general['pastures'][pinp.general['pas_inc']]):
                pasture_dse_carry[pasture] = pinp.sheep['i_sr_constraint_t'][t]
            ###constraint
            try:
                model.del_component(model.con_SR_bound)
                model.del_component(model.con_SR_bound_index)
            except AttributeError:
                pass
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
        if landuse_bound_inc:
            ##initilise bound - note zero is the equivalent of no bound
            landuse_bound_k = pd.Series(0,index=model.s_landuses) #use landuse2 because that is the expanded version of pasture phases eg t, tr not just tedera
            ##set bound - note that setting to zero is the equivalent of no bound
            landuse_bound_k.iloc[0] = 50
            ###dict
            landuse_area_bound = dict(landuse_bound_k)
            ###constraint
            try:
                model.del_component(model.con_landuse_bound)
            except AttributeError:
                pass
            def k_bound(model, k):
                if landuse_area_bound[k]!=0:  #bound will not be built if param == 0
                    return(
                           sum(model.v_phase_area[r, l] * model.p_landuse_area[r, k] for r in model.s_phases for l in model.s_lmus for t in model.s_pastures)
                           == landuse_area_bound[k])
                else:
                    pe.Constraint.Skip
            model.con_pas_bound = pe.Constraint(model.s_landuses, rule=k_bound, doc='bound on total pasture area')

        ##total pasture area - hence also total crop area
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



 


