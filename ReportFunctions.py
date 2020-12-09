# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 09:58:05 2020

@author: young

This module should not import inputs (incase the inputs are adjusted during the exp so they will not be correct for r_valsing)
When creating r_vals values try and do it in obvious spots even if you need to go out of the way to do it eg phases in rotation.py
"""

import pandas as pd
import numpy as np

import Functions as fun
na=np.newaxis



#################
# Final reports #
#################
def f_errors(r_vals, exp_data, trial_outdated, trials):
    ##first check if data exists for each desired trial
    try:
        for row in trials:
            r_vals[exp_data.index[row][2]]
        status = 'good'
    except:
        print('''

              reporting for trials that dont exist    

              ''')
        status='bad'  # exit function
    ##second check if generating results using out of date data.
    if any(trial_outdated[trials]):
        print('''

              Generating reports from out dated data

              ''')
    return status

def pnl(lp_vars, r_vals, exp_data, trial_outdated, trials):
    '''Returns profit and loss statement for selected trials. Multiple trials result in a stacked pnl table'''
    ##check for errors
    status = f_errors(r_vals, exp_data, trial_outdated, trials)
    if status=='bad':
        return #exit function if data doesnt exist


##between trial functions
###could use automatic trial selection.
rep.f_across_trials_summary(inter, trials, criterea)
##create intermidiates for each trial
for row in range(len(exp_data)):
    ##check to make sure user wants to run this trial
    if exp_data.index[row][0] == True:
        inter[exp_data.index[row][2]]={}
        rep.intermediates(inter[exp_data.index[row][2]], r_vals[exp_data.index[row][2]], lp_vars[exp_data.index[row][2]])
        ##within trial reports
        rep.f_dse(inter[exp_data.index[row][2]])
        rep.f_profitloss_table(inter[exp_data.index[row][2]])




#########################################
# intermidiate report building functions#
#########################################



def intermediates(inter, r_vals, lp_vars):
    '''

    Parameters
    ----------
    inter : Dict
        Pass in a dict to store intermidiate values.
    r_vals : Dict
        Pass in dict with all r_vals values from precalcs.
    lp_vals : Dict
        Pass in dict with lp variable results from pyomo.

    Returns
    -------
    Here we manipulate r_vals variables and combine with the lp result variables.
    Everything is stored in the 'inter' dict which is passed into each
    function which preforms any niche calculations & builds the relevant table or figure.

    There are two options
        1.convert the dict to pandas dataframe: see crop and pasture area
        2.convert dict to numpy: see dse

    '''
    ##keys
    keys_c = r_vals['fin']['keys_c']
    keys_a = r_vals['stock']['keys_a']
    keys_d = r_vals['stock']['keys_d']
    keys_g0 = r_vals['stock']['keys_g0']
    keys_g1 = r_vals['stock']['keys_g1']
    keys_g2 = r_vals['stock']['keys_g2']
    keys_g3 = r_vals['stock']['keys_g3']
    keys_f = r_vals['stock']['keys_f']
    keys_h1 = r_vals['stock']['keys_h1']
    keys_i = r_vals['stock']['keys_i']
    keys_k2 = r_vals['stock']['keys_k2']
    keys_k3 = r_vals['stock']['keys_k3']
    keys_k5 = r_vals['stock']['keys_k5']
    keys_lw1 = r_vals['stock']['keys_lw1']
    keys_lw3 = r_vals['stock']['keys_lw3']
    keys_lw_prog = r_vals['stock']['keys_lw_prog']
    keys_n1 = r_vals['stock']['keys_n1']
    keys_n3 = r_vals['stock']['keys_n3']
    keys_p8 = r_vals['stock']['keys_p8']
    keys_t1 = r_vals['stock']['keys_t1']
    keys_t2 = r_vals['stock']['keys_t2']
    keys_t3 = r_vals['stock']['keys_t3']
    keys_v1 = r_vals['stock']['keys_v1']
    keys_v3 = r_vals['stock']['keys_v3']
    keys_y0 = r_vals['stock']['keys_y0']
    keys_y1 = r_vals['stock']['keys_y1']
    keys_y3 = r_vals['stock']['keys_y3']
    keys_x = r_vals['stock']['keys_x']
    keys_z = r_vals['stock']['keys_z']
    keys_p6 = r_vals['stock']['keys_p6']
    keys_p5 = r_vals['lab']['keys_p5']
    keys_pastures = r_vals['pas']['keys_pastures']

    inter['keys_p6'] = keys_p6
    inter['keys_c'] = keys_c


    ##axis len
    len_c = len(keys_c)
    len_a = len(keys_a)
    len_d = len(keys_d)
    len_g0 = len(keys_g0)
    len_g1 = len(keys_g1)
    len_g2 = len(keys_g2)
    len_g3 = len(keys_g3)
    len_f = len(keys_f)
    len_h1 = len(keys_h1)
    len_i = len(keys_i)
    len_k2 = len(keys_k2)
    len_k3 = len(keys_k3)
    len_k5 = len(keys_k5)
    len_lw1 = len(keys_lw1)
    len_lw3 = len(keys_lw3)
    len_lw_prog = len(keys_lw_prog)
    len_n1 = len(keys_n1)
    len_n3 = len(keys_n3)
    len_p8 = len(keys_p8)
    len_t1 = len(keys_t1)
    len_t2 = len(keys_t2)
    len_t3 = len(keys_t3)
    len_v1 = len(keys_v1)
    len_v3 = len(keys_v3)
    len_y0 = len(keys_y0)
    len_y1 = len(keys_y1)
    len_y3 = len(keys_y3)
    len_x = len(keys_x)
    len_z = len(keys_z)
    len_p6 = len(keys_p6)
    len_p5 = len(keys_p5)


def f_rotation_landuse_summary(lp_vars, r_vals, inter, option=0):
    '''
    Rotation summary. With multiple output levels.
    return options:
    0- tuple: all results wraped in tuple
    0- table: all rotations by lmu
    1- table: selected rotations by lmu
    2- table: crop and pasture area by lmu
    3- float: total pasture area
    4- float: total crop area

    Note: variable stored in inter dict are accesible in the between trial functions
    '''

    ##landuse sets
    all_pas = r_vals['rot']['all_pastures']
    ##rotation
    phases_df = r_vals['rot']['phases']
    phases_rk = phases_df.set_index(5, append=True)  # add landuse as index level
    rot_area_rl = pd.Series(lp_vars['v_phase_area']) #create a series of all the phase areas
    rot_area_rkl = rot_area_rl.unstack().reindex(phases_rk.index, axis=0, level=0).stack() #add landuse to the axis
    landuse_area_kl = rot_area_rkl.sum(axis=0,level=(1,2)).unstack() #area of each landuse (sum lmu and rotation)
    ##all rotations by lmu
    rot_area_rl = rot_area_rl.unstack()
    if option==1:
        return rot_area_rl
    ##selected rotations by lmu
    rot_area_selected_rl = rot_area_rl[any(rot_area_rl,axis=1)]
    if option==2:
        return rot_area_selected_rl
    ###crop & pasture area
    ####you can now use isin pasture or crop sets to calc the area of crop or pasture
    total_pasture_area_l = landuse_area_kl[landuse_area_kl.index.isin(all_pas)].sum()
    total_crop_area_l = landuse_area_kl[~landuse_area_kl.index.isin(all_pas)].sum()
    ##crop & pasture area by lmu
    croppas_area = pd.DataFrame()
    croppas_area.loc['pasture'] = inter['pasture_area_l']
    croppas_area.loc['crop'] = inter['crop_area_l']
    if option==3:
        return croppas_area
    ##store any values which are used in the between trial reports
    inter['rot_area_rl'] = rot_area_rl
    inter['pasture_area_l'] = total_pasture_area_l
    inter['crop_area_l'] = total_crop_area_l
    inter['pasture_area'] = total_pasture_area_l.sum()
    inter['crop_area'] = total_crop_area_l.sum()


    ##mach
    contractharv_hours = pd.Series(lp_vars['v_contractharv_hours'])
    harv_hours = pd.Series(lp_vars['v_harv_hours']).sum(level=1) #sum labour axis
    harvest_cost = r_vals['mach']['contract_harvest_cost'].mul(contractharv_hours, axis=1)  + r_vals['mach']['harvest_cost'].mul(harv_hours, axis=1)
    seeding_days = pd.Series(lp_vars['v_seeding_machdays']).sum(level=(1,2)) #sum labour period axis
    contractseeding_ha = pd.Series(lp_vars['v_contractseeding_ha']).sum(level=1) #sum labour period and lmu axis
    seeding_ha = r_vals['mach']['seeding_rate'].mul(seeding_days.unstack()).stack() #note seeding ha wont equal the rotation area because arable area is included in seed_ha.
    seeding_cost_own = r_vals['mach']['seeding_cost'].reindex(seeding_ha.index, axis=1,level=1).mul(seeding_ha,axis=1).sum(axis=1, level=0) #sum lmu axis
    contractseed_cost_ha = r_vals['mach']['contractseed_cost']
    idx = pd.MultiIndex.from_product([contractseed_cost_ha.index, contractseeding_ha.index])
    seeding_cost_contract = contractseed_cost_ha.reindex(idx, level=0).mul(contractseeding_ha, level=1).unstack()
    exp_mach_ha_rlc = pd.concat([r_vals['crop']['fert_app_cost'], r_vals['crop']['nap_fert_app_cost'], r_vals['crop']['chem_app_cost_ha']],axis=1).sum(axis=1,level=0) #cost per ha
    exp_mach_kc = exp_mach_ha_rlc.unstack().reindex(phases_rk.index,axis=0,level=0).stack().mul(rot_area_rkl,axis=0).sum(axis=0,level=1) #reindex to include landuse then mul area and sum lmu and rot
    exp_mach_kc = pd.concat([exp_mach_kc.T, seeding_cost_own, seeding_cost_contract, harvest_cost],axis=0).sum(axis=0,level=0).T
    inter['mach_exp'] = exp_mach_kc

    ##cropping
    ###expenses
    exp_crop_fert_ha = pd.concat([r_vals['crop']['phase_fert_cost'], r_vals['crop']['nap_phase_fert_cost']],axis=1).sum(axis=1,level=0)
    exp_crop_fert = exp_crop_fert_ha.unstack().reindex(phases_rk.index,axis=0,level=0).stack().mul(rot_area_rkl,axis=0).sum(axis=0,level=1) #reindex to include landuse then mul area and sum lmu and rot
    exp_crop_chem = r_vals['crop']['chem_cost'].unstack().reindex(phases_rk.index,axis=0,level=0).stack().mul(rot_area_rkl,axis=0).sum(axis=0,level=1) #reindex to include landuse then mul area and sum lmu and rot
    misc_cropping_exp_ha = pd.concat([r_vals['crop']['stub_cost'], r_vals['crop']['insurance_cost'], r_vals['crop']['seedcost']],axis=1).sum(axis=1,level=0) #stubble, seed & insurance
    misc_cropping_exp = misc_cropping_exp_ha.unstack().reindex(phases_rk.index,axis=0,level=0).stack().mul(rot_area_rkl,axis=0).sum(axis=0,level=1) #reindex to include landuse then mul area and sum lmu and rot
    inter['fert_exp'] = exp_crop_fert
    inter['chem_exp'] = exp_crop_chem
    inter['misc_exp'] = misc_cropping_exp
    ###revenue. rev = (grain_sold + grain_fed - grain_purchased) * sell_price
    grain_purchased = pd.Series(lp_vars['v_buy_grain'])
    grain_sold = pd.Series(lp_vars['v_sell_grain'])
    grain_fed_kg = pd.Series(lp_vars['v_sup_con']).sum(level=(0,1)) #sum feed pool and feed period
    grain_fed_kp5 = pd.Series(lp_vars['v_sup_con']).sum(level=(0,3)).swaplevel() #sum feed pool and grain pool
    total_grain = grain_sold + grain_fed_kg - grain_purchased #total grain produced by crop enterprise
    grains_sale_price = r_vals['crop']['grain_price'].T.stack()
    grain_rev = grains_sale_price.mul(total_grain.reindex(grains_sale_price.index), axis=0).sum(axis=0,level=0) #sum grain pool, have to reindex (not really sure why since it is the same index - maybe one has been condensed ie index with nan removed)
    inter['grain_rev'] = grain_rev

    ##stock
    ###animal numbers
    sire_shape = len_g0
    dams_shape = len_k2, len_t1, len_v1, len_a, len_n1, len_lw1, len_z, len_i, len_y1, len_g1
    offs_shape = len_k3, len_k5, len_t3, len_v3, len_n3, len_lw3, len_z, len_i, len_a, len_x, len_y3, len_g3
    sire_numbers = np.array(list(lp_vars['v_sire'].values()))
    sire_numbers_g0 = sire_numbers.reshape(sire_shape)
    sire_numbers_g0[sire_numbers_g0==None] = 0 #replace None with 0
    inter['sire_numbers_g0'] = sire_numbers_g0
    dam_numbers = np.array(list(lp_vars['v_dams'].values()))
    dam_numbers_k2tvanwziy1g1 = dam_numbers.reshape(dams_shape)
    dam_numbers_k2tvanwziy1g1[dam_numbers_k2tvanwziy1g1==None] = 0 #replace None with 0
    inter['dam_numbers_k2tvanwziy1g1'] = dam_numbers_k2tvanwziy1g1
    offs_numbers = np.array(list(lp_vars['v_offs'].values()))
    offs_numbers_k3k5tvnwziaxy1g1 = offs_numbers.reshape(offs_shape)
    offs_numbers_k3k5tvnwziaxy1g1[offs_numbers_k3k5tvnwziaxy1g1==None] = 0 #replace None with 0
    inter['offs_numbers_k3k5tvnwziaxy1g1'] = offs_numbers_k3k5tvnwziaxy1g1
    ###dse
    siredse_shape = len_p6, len_g0
    damsdse_shape = len_k2, len_p6, len_t1, len_v1, len_a, len_n1, len_lw1, len_z, len_i, len_y1, len_g1
    offsdse_shape = len_k3, len_k5, len_p6, len_t3, len_v3, len_n3, len_lw3, len_z, len_i, len_a, len_x, len_y3, len_g3
    inter['dsenw_p6g0'] = r_vals['stock']['dsenw_p6g0'].reshape(siredse_shape)
    inter['dsenw_k2p6tva1nwziyg1'] = r_vals['stock']['dsenw_k2p6tva1nwziyg1'].reshape(damsdse_shape)
    inter['dsenw_k3k5p6tvnwzixyg3'] = r_vals['stock']['dsenw_k3k5p6tvnwzixyg3'].reshape(offsdse_shape)
    inter['dsemj_p6g0'] = r_vals['stock']['dsemj_p6g0'].reshape(siredse_shape)
    inter['dsemj_k2p6tva1mjziyg1'] = r_vals['stock']['dsemj_k2p6tva1nwziyg1'].reshape(damsdse_shape)
    inter['dsemj_k3k5p6tvmjzixyg3'] = r_vals['stock']['dsemj_k3k5p6tvnwzixyg3'].reshape(offsdse_shape)
    ###expenses sup feeding
    grains_buy_price = r_vals['sup']['buy_grain_price'].T.stack()
    grain_exp = (grains_sale_price.mul((grain_fed_kg - grain_purchased).reindex(grains_sale_price.index), axis=0)
                + grains_buy_price.mul(grain_purchased.reindex(grains_sale_price.index), axis=0)).sum(axis=0,level=0) #sum grain pool
    feeding_exp_kp5c = r_vals['sup']['total_sup_cost'] #feeding and storage cost related to sup per tonne, sum fp axis
    feeding_exp =  feeding_exp_kp5c.mul(grain_fed_kp5,axis=0).sum(axis=0, level=0)
    ###husbandry expense
    sirecost_shape = len_c, len_g0
    damscost_shape = len_k2, len_c, len_t1, len_v1, len_a, len_n1, len_lw1, len_z, len_i, len_y1, len_g1
    offscost_shape = len_k3, len_k5, len_c, len_t3, len_v3, len_n3, len_lw3, len_z, len_i, len_a, len_x, len_y3, len_g3

    inter['sire_cost'] = r_vals['stock']['cost_cg0'].reshape(sirecost_shape) * sire_numbers_g0
    inter['dams_cost'] = r_vals['stock']['cost_k2ctva1nwziyg1'].reshape(damscost_shape) * dam_numbers_k2tvanwziy1g1[:,na,...]
    inter['offs_cost'] = r_vals['stock']['cost_k3k5ctvnwzixyg3'].reshape(offscost_shape) * offs_numbers_k3k5tvnwziaxy1g1[:,:,na,...]
    ###sale income
    inter['sire_sale'] = r_vals['stock']['salevalue_cg0'].reshape(sirecost_shape) * sire_numbers_g0
    inter['dams_sale'] = r_vals['stock']['salevalue_k2ctva1nwziyg1'].reshape(damscost_shape) * dam_numbers_k2tvanwziy1g1[:,na,...]
    inter['offs_sale'] = r_vals['stock']['salevalue_k3k5ctvnwzixyg3'].reshape(offscost_shape) * offs_numbers_k3k5tvnwziaxy1g1[:,:,na,...]
    ###wool income
    inter['sire_wool'] = r_vals['stock']['woolvalue_cg0'].reshape(sirecost_shape) * sire_numbers_g0
    inter['dams_wool'] = r_vals['stock']['woolvalue_k2ctva1nwziyg1'].reshape(damscost_shape) * dam_numbers_k2tvanwziy1g1[:,na,...]
    inter['offs_wool'] = r_vals['stock']['woolvalue_k3k5ctvnwzixyg3'].reshape(offscost_shape) * offs_numbers_k3k5tvnwziaxy1g1[:,:,na,...]


    ##labour
    inter['cas_cost_pc'] = r_vals['lab']['casual_cost'].mul(pd.Series(lp_vars['v_quantity_casual']),level=0)
    inter['perm_cost_c'] = r_vals['lab']['perm_cost'] * pd.Series(lp_vars['v_quantity_perm']).values
    inter['manager_cost_c'] = r_vals['lab']['manager_cost'] * pd.Series(lp_vars['v_quantity_manager']).values


    ##dep - depreciation is yearly but for the profit and loss it is equally divided into each cash period
    dep = lp_vars['v_dep'][None]/len_c #convert to dep per cashflow period
    inter['dep_c'] = pd.Series([dep]*len_c, index=keys_c)  #convert to df with cashflow period as index
    ##overheads/fixed expenses
    inter['exp_fix_c'] = r_vals['fin']['overheads']




    # df_rot = df_rot.rename_axis(['rot','lmu'])
    # phase_area = pd.merge(r_vals['rot']['phases'], df_rot, how='left', left_index=True, right_on=['rot']) #merge full phase array with area array
    # phase_is_pasture = phase_area.iloc[:,-2].isin(r_vals['rot']['all_pastures'])
    # inter['pasture_area'] = df_rot[phase_is_pasture].sum()
    # pasture_area_rt = pd.DataFrame(r_vals['pas']['pasture_area_rt'], index=phases_df.index, columns=keys_pastures)
    # inter['pasture_area'] = pasture_area_rt.mul(rot_area,axis=0,level=0).sum(axis=0) #return the area of each pasture type
    # inter['crop_area'] = df_rot[~phase_is_pasture].sum() #^do i have something like pasture already? or do i need to do option 1? how can i get area for each crop set?








def f_make_table(data, index, header):
    '''function to return table
    ^currently just returns a df but there are python packages which make nice tables'''
    return pd.DataFrame(data, index=index, columns=header)

def f_dse(inter,method=0,per_ha=False):
    '''

    :param
    inter: dict
    method: int
            0 - dse by normal weight
            1 - dse by mei
    per_ha: Bool
        if true it returns DSE/ha else it returns total dse
    :return DSE per pasture hectare for each sheep group:
    '''
    if method==0:
        ##sire
        dse_sire = inter['sire_numbers_g0'] * inter['dsenw_p6g0']
        ##dams
        dse_dams = fun.f_reduce_skipfew(np.sum, inter['dam_numbers_k2tvanwziy1g1'][:,na,...] * inter['dsenw_k2p6tva1nwziyg1'], preserveAxis=1) #sum all axis except p6
        ##dams
        dse_offs = fun.f_reduce_skipfew(np.sum, inter['offs_numbers_k3k5tvnwziaxy1g1'][:,:,na,...] * inter['dsenw_k3k5p6tvnwzixyg3'], preserveAxis=2) #sum all axis except p6
    else:
        ##sire
        dse_sire = inter['sire_numbers_g0'] * inter['dsemj_p6g0']
        ##dams
        dse_dams = fun.f_reduce_skipfew(np.sum, inter['dam_numbers_k2tvanwziy1g1'][:,na,...] * inter['dsemj_k2p6tva1nwziyg1'], preserveAxis=1) #sum all axis except p6
        ##dams
        dse_offs = fun.f_reduce_skipfew(np.sum, inter['offs_numbers_k3k5tvnwziaxy1g1'][:,:,na,...] * inter['dsemj_k3k5p6tvnwzixyg3'], preserveAxis=2) #sum all axis except p6

    ##dse per ha if user opts for this level of detail
    if per_ha:
        dse_sire = dse_sire/inter['pasture_area']
        dse_dams = dse_dams/inter['pasture_area']
        dse_offs = dse_offs/inter['pasture_area']

    ##turn to table
    dse_sire = f_make_table(dse_sire, inter['keys_p6'], ['Sire DSE'])
    dse_dams = f_make_table(dse_dams, inter['keys_p6'], ['Dams DSE'])
    dse_offs = f_make_table(dse_offs, inter['keys_p6'], ['Offs DSE'])
    return dse_sire, dse_dams, dse_offs

def f_profitloss_table(inter):
    '''

    Parameters
    ----------
    inter : Dict
        Pass in dict with all intermidiate values required to calculate p/l.

    Returns
    -------
    r_vals (table or figure etc).

    '''
    ##create p/l dataframe
    pnl_index = pd.MultiIndex(levels=[[], []],
                             codes=[[], []],
                             names=['Type', 'Subtype'])
    pnl = pd.DataFrame(index=pnl_index, columns=inter['keys_c']) #need to initilise df with multiindex so rows can be added
    ##income
    ###sum axis to return total income in each cash peirod
    siresale_c = fun.f_reduce_skipfew(np.sum, inter['sire_sale'], preserveAxis=0) #sum all axis except c
    damssale_c = fun.f_reduce_skipfew(np.sum, inter['dams_sale'], preserveAxis=1) #sum all axis except c
    offssale_c = fun.f_reduce_skipfew(np.sum, inter['offs_sale'], preserveAxis=2) #sum all axis except c
    sirewool_c = fun.f_reduce_skipfew(np.sum, inter['sire_wool'], preserveAxis=0) #sum all axis except c
    damswool_c = fun.f_reduce_skipfew(np.sum, inter['dams_wool'], preserveAxis=1) #sum all axis except c
    offswool_c = fun.f_reduce_skipfew(np.sum, inter['offs_wool'], preserveAxis=2) #sum all axis except c
    stocksale_c = siresale_c + damssale_c + offssale_c
    wool_c = sirewool_c + damswool_c + offswool_c
    grain_c = inter['grain_rev'].sum(axis=0) #sum landuse axis
    ###add to p/l table each as a new row
    pnl.loc[('Revenue', 'grain'),:] = grain_c
    pnl.loc[('Revenue', 'sheep sales'),:] = stocksale_c
    pnl.loc[('Revenue', 'wool'),:] = wool_c
    pnl.loc[('Revenue', 'Total Revenue'),:] = pnl.loc[pnl.index.get_level_values(0) == 'Revenue'].sum(axis=0)

    ##expenses
    ###sum axis to return total cost in each cash peirod
    ####stock
    sirecost_c = fun.f_reduce_skipfew(np.sum, inter['sire_cost'], preserveAxis=0) #sum all axis except c
    damscost_c = fun.f_reduce_skipfew(np.sum, inter['dams_cost'], preserveAxis=1) #sum all axis except c
    offscost_c = fun.f_reduce_skipfew(np.sum, inter['offs_cost'], preserveAxis=2) #sum all axis except c
    stockcost_c = sirecost_c + damscost_c + offscost_c
    ####machinery
    mach_c = inter['mach_exp'].sum(axis=0) #sum landuse
    ####crop & pasture
    pasfert_c = inter['fert_exp'][inter['fert_exp'].index.isin(inter['pas_set'])].sum(axis=0)
    cropfert_c = inter['fert_exp'][~inter['fert_exp'].index.isin(inter['pas_set'])].sum(axis=0)
    paschem_c = inter['chem_exp'][inter['chem_exp'].index.isin(inter['pas_set'])].sum(axis=0)
    cropchem_c = inter['chem_exp'][~inter['chem_exp'].index.isin(inter['pas_set'])].sum(axis=0)
    pasmisc_c = inter['misc_exp'][inter['misc_exp'].index.isin(inter['pas_set'])].sum(axis=0)
    cropmisc_c = inter['misc_exp'][~inter['misc_exp'].index.isin(inter['pas_set'])].sum(axis=0)
    pas_c = pasfert_c + paschem_c + pasmisc_c
    crop_c = cropfert_c + cropchem_c + cropmisc_c
    ####labour
    labour_c = inter['cas_cost_pc'].sum(level=1) + inter['perm_cost_c'] + inter['manager_cost_c']
    ###add to p/l table each as a new row
    pnl.loc[('Expense', 'Crop'),:] = crop_c
    pnl.loc[('Expense', 'pasture'),:] = pas_c
    pnl.loc[('Expense', 'stock'),:] = stockcost_c
    pnl.loc[('Expense', 'machinery'),:] = mach_c
    pnl.loc[('Expense', 'labour'),:] = labour_c
    pnl.loc[('Expense', 'fixed'),:] = inter['exp_fix_c']
    pnl.loc[('Expense', 'depreciation'),:] = inter['dep_c']
    pnl.loc[('Expense', 'Total expenses'),:] = pnl.loc[pnl.index.get_level_values(0) == 'Expense'].sum(axis=0)

    ##EBIT
    pnl.loc[('', 'EBIT')] = pnl.loc[('Revenue', 'Total Revenue')] - pnl.loc[('Expense', 'Total expenses')]

    ##add a column which is total of all casflow period
    pnl['Full year'] = pnl.sum(axis=1)

    ##round numbers in df
    pnl = pnl.round(1)
    return pnl

def f_landuse_summary(inter, option=0):
    '''returns 3 tables:
    0- all rotations by lmu
    1- selected rotations by lmu
    2- crop and pasture area by lmu
    '''
    ##all rotations by lmu
    rot_area_rl = inter['rot_area_rl'].unstack()
    if option==0:
        return rot_area_rl
    ##selected rotations by lmu
    rot_area_selected_rl = rot_area_rl[any(rot_area_rl,axis=1)]
    if option==1:
        return rot_area_selected_rl
    ##crop & pasture area by lmu
    croppas_area = pd.DataFrame()
    croppas_area.loc['pasture'] = inter['pasture_area_l']
    croppas_area.loc['crop'] = inter['crop_area_l']
    if option==2:
        return croppas_area

def f_labour_summary(inter):
    '''Labour summary
    0- quantity of each labour source in each period
    1- yearly labour cost for each labour source
    '''

def f_input_summary_prices(inter):
    '''summary of the inputs used in a trial - this is reported after SA has been applied'''


def f_across_trials_summary(inter):
    '''Returns results summary for all of the trials in exp'''
