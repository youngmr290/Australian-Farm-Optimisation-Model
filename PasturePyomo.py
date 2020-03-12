# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 15:00:25 2019

@author: John
"""
#python modules
from pyomo.environ import *

#MUDAS modules
from CreateModel import *
import Pasture as pas

pastures = uinp.structure['pastures']       

# Conservation limit over feed period(i) & soil type(j) = 50 constraints
# Note: the MIDAS version was effectively just 5 constraints achieved
# by subtracting the conservation limit from the period 1 low quality Dry foo row (and relying on non negative FOO)
# sum(sum(foo(f,l)>=erosion_limit(f,l) for f in feed_period) for l in lmu)




#####################################################################################################################################################################################################
#####################################################################################################################################################################################################
# Params
#####################################################################################################################################################################################################
#####################################################################################################################################################################################################

pas.init_and_map_excel('Property.xlsx', pastures)                         # read inputs from Excel file and map to the python variables
pas.calculate_germ_and_reseed()                          # calculate the germination for each rotation phase
pas.green_and_dry()                            # calculate the FOO lost when destocked and the FOO gained when grazed after establishment

try:
    model.del_component(model.p_germination)
except AttributeError:
    pass
model.p_germination = Param(model.s_feed_periods, model.s_lmus, model.s_phases, model.s_pastures, initialize=pas.p_germination_flrt, default=0, doc='pasture germination for each rotation')

try:
    model.del_component(model.p_germination)
except AttributeError:
    pass
model.p_foo_grn_reseeding = Param(model.s_feed_periods, model.s_lmus, model.s_phases, model.s_pastures, initialize=pas.p_foo_grn_reseeding_flrt, default=0, doc='Change in grn FOO due to destocking and seeding of pastures')

try:
    model.del_component(model.p_germination)
except AttributeError:
    pass
model.p_foo_dry_reseeding = Param(model.s_dry_groups, model.s_feed_periods, model.s_lmus, model.s_phases, model.s_pastures, initialize=pas.p_foo_dry_reseeding_dflrt, default=0, doc='Change in dry FOO due to destocking and seeding of pastures')

try:
    model.del_component(model.p_foo_end_grnha)
except AttributeError:
    pass
model.p_foo_end_grnha = Param(model.s_grazing_int, model.s_foo_levels, model.s_feed_periods, model.s_lmus, model.s_pastures, initialize=pas.p_foo_end_grnha_goflt, default=0, doc='Green Foo at the end of the period')

try:
    model.del_component(model.p_foo_start_grnha)
except AttributeError:
    pass
model.p_foo_start_grnha = Param(model.s_foo_levels, model.s_feed_periods, model.s_lmus, model.s_pastures, initialize=pas.p_foo_start_grnha_oflt, default=0, doc='Green Foo at the start of the period')

try:
    model.del_component(model.p_senesce_grnha)
except AttributeError:
    pass
model.p_senesce_grnha = Param(model.s_dry_groups, model.s_grazing_int, model.s_foo_levels, model.s_feed_periods, model.s_lmus, model.s_pastures, initialize=pas.p_senesce_grnha_dgoflt, default=0, doc='Green pasture senescence into high and low quality dry pasture pools')

try:
    model.del_component(model.p_me_cons_grnha)
except AttributeError:
    pass
model.p_me_cons_grnha = Param(model.s_sheep_pools, model.s_grazing_int, model.s_foo_levels, model.s_feed_periods, model.s_lmus, model.s_pastures, initialize=pas.p_me_cons_grnha_egoflt, default=0, doc='Total ME from grazing a hectare')

try:
    model.del_component(model.p_volume_grnha)
except AttributeError:
    pass
model.p_volume_grnha = Param(model.s_sheep_pools, model.s_grazing_int, model.s_foo_levels, model.s_feed_periods, model.s_lmus, model.s_pastures, initialize=pas.p_volume_grnha_egoflt, default=0, doc='Total Vol from grazing a hectare')

try:
    model.del_component(model.p_dry_mecons_t)
except AttributeError:
    pass
model.p_dry_mecons_t = Param(model.s_sheep_pools, model.s_dry_groups, model.s_feed_periods, model.s_pastures, initialize=pas.p_dry_mecons_t_edft, default=0, doc='Total ME from grazing a tonne of dry feed')

try:
    model.del_component(model.p_dry_volume_t)
except AttributeError:
    pass
model.p_dry_volume_t = Param(model.s_dry_groups, model.s_feed_periods, model.s_pastures, initialize=pas.p_dry_volume_t_dft, default=0, doc='Total Vol from grazing a tonne of dry feed')

try:
    model.del_component(model.p_dry_transfer_t)
except AttributeError:
    pass
model.p_dry_transfer_t = Param(model.s_dry_groups, model.s_feed_periods, model.s_pastures, initialize=pas.p_dry_transfer_t_dft, default=0, doc='quantity of dry feed transfered out of the period to the next')

try:
    model.del_component(model.p_dry_removal_t)
except AttributeError:
    pass
model.p_dry_removal_t = Param(model.s_dry_groups, model.s_feed_periods, model.s_pastures, initialize=pas.p_dry_removal_t_dft, default=0, doc='quantity of dry feed removed for sheep to consume 1t')

try:
    model.del_component(model.p_nap)
except AttributeError:
    pass
model.p_nap = Param(model.s_dry_groups, model.s_lmus, model.s_pastures, initialize=pas.p_nap_dlt, default=0, doc='non arable pasture')

try:
    model.del_component(model.p_phase_area)
except AttributeError:
    pass
model.p_phase_area = Param(model.s_feed_periods, model.s_phases, model.s_pastures, initialize=pas.p_phase_area_frt, default=0, doc='pasture area in each rotation for each feed period')

try:
    model.del_component(model.p_pas_sow)
except AttributeError:
    pass
model.p_pas_sow = Param(model.s_periods, model.s_lmus, model.s_phases, model.s_landuses, initialize=pas.p_pas_sow_plrt, default=0, doc='pasture sown for each rotation')

try:
    model.del_component(model.p_poc_con)
except AttributeError:
    pass
model.p_poc_con = Param(model.s_feed_periods ,model.s_lmus, initialize=pas.poc_con(),default=0, doc='consumption of pasture on 1ha of a crop paddock each day for each lmu in each feed period')

try:
    model.del_component(model.p_poc_md)
except AttributeError:
    pass
model.p_poc_md = Param(model.s_feed_periods, initialize=pas.poc_md(),default=0, doc='md of pasture on crop paddocks for each feed period')

try:
    model.del_component(model.p_poc_vol)
except AttributeError:
    pass
model.p_poc_vol = Param(model.s_feed_periods, initialize=pas.poc_vol(),default=0, doc='vol (ri intake) of pasture on crop paddocks for each feed period')

#####################################################################################################################################################################################################
#####################################################################################################################################################################################################
# Local constraints
#####################################################################################################################################################################################################
#####################################################################################################################################################################################################
def greenpas(model,p,l):
    return model.v_phase_area[r,l] * model.p_germination[f,l,r,k]
con_greenpas = Constraint(model.s_periods, model.s_lmus, rule = greenpas, doc='green pasture available on each soil type in each feed period')

#####################################################################################################################################################################################################
#####################################################################################################################################################################################################
# Variables
#####################################################################################################################################################################################################
#####################################################################################################################################################################################################

model.v_greenpas_ha = Var(model.s_sheep_pools, model.s_grazing_int, model.s_foo_levels, model.s_feed_periods, model.s_lmus, model.s_pastures, bounds = (0,None) , doc='hectares grazed each period for each grazing intensity on each soil in each period')
model.v_drypas_consumed = Var(model.s_sheep_pools, model.s_dry_groups, model.s_feed_periods, model.s_pastures, bounds = (0,None) , doc='tonnes of low and high quality dry feed consumed by each sheep pool in each feed period')
model.v_drypas_transfer = Var(model.s_dry_groups, model.s_feed_periods, model.s_pastures, bounds = (0,None) , doc='tonnes of low and high quality dry feed transferred to the following periods in each feed period')
model.v_poc = Var(model.s_feed_periods, model.s_lmus, model.s_sheep_pools, bounds = (0,None) , doc='tonnes of poc consumed by each sheep pool in each period on each lmu')



#####################################################################################################################################################################################################
#####################################################################################################################################################################################################
# Functions for coremodel
#####################################################################################################################################################################################################
#####################################################################################################################################################################################################

##############
#sow         #
##############
def cropsow(model,p,k,l):
    return sum(model.p_pas_sow[p,l,r,k]*model.v_phase_area[r,l] for r in model.s_phases if model.p_pas_sow[p,l,r,k] != 0) 













