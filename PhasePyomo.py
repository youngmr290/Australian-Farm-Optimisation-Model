
"""

author: young


"""

#python modules
import pyomo.environ as pe
import timeit

#AFO modules
import Phase as phs
import PropertyInputs as pinp

def crop_precalcs(params, r_vals):
    '''
    Call crop precalc functions.

    :param params: dictionary which stores all arrays used to populate pyomo parameters.
    :param report: dictionary which stores all report values.

    '''

    phs.f1_crop_params(params, r_vals)


def f1_croppyomo_local(params, model):
    ''' Builds pyomo variables, parameters and constraints'''

    ############
    # variable #
    ############
    model.v_sell_grain = pe.Var(model.s_phase_periods, model.s_season_types, model.s_crops, model.s_grain_pools, bounds=(0,None),
                                doc='tonnes of grain in each pool sold')

    model.v_grain_debit = pe.Var(model.s_phase_periods, model.s_season_types, model.s_crops, model.s_grain_pools, bounds=(0,None),
                                doc='tonnes of grain in debt (will need to be purchased or provided from harvest)')

    model.v_grain_credit = pe.Var(model.s_phase_periods, model.s_season_types, model.s_crops, model.s_grain_pools, bounds=(0,None),
                                doc='tonnes of grain in credit (can be used for sup feeding or sold)')

    #########
    #param  #
    #########

    model.p_rotation_cost = pe.Param(model.s_enterprises, model.s_season_periods, model.s_season_types, model.s_phase_periods, model.s_lmus, model.s_phases, initialize=params['rot_cost'], default=0, mutable=False, doc='total cost for 1 unit of rotation')
       
    model.p_increment_rotation_cost = pe.Param(model.s_enterprises, model.s_season_periods, model.s_season_types, model.s_lmus,
                                               model.s_phases, model.s_phase_periods, initialize=params['increment_rot_cost'],
                                               default=0, mutable=False, doc='total cost for 1 unit of rotation')

    model.p_rotation_wc = pe.Param(model.s_enterprises, model.s_season_periods, model.s_season_types, model.s_phase_periods,
                                   model.s_lmus, model.s_phases, initialize=params['rot_wc'], default=0, mutable=False,
                                   doc='total wc for 1 unit of rotation')
       
    model.p_increment_rotation_wc = pe.Param(model.s_enterprises, model.s_season_periods, model.s_season_types, model.s_lmus,
                                             model.s_phases, model.s_phase_periods, initialize=params['increment_rot_wc'],
                                             default=0, mutable=False, doc='total wc for 1 unit of rotation')

    model.p_rotation_yield = pe.Param(model.s_phases, model.s_crops, model.s_lmus, model.s_season_types, model.s_phase_periods,
                                      initialize=params['rot_yield'], default = 0.0, mutable=False, doc='grain production for all crops for 1 unit of rotation')

    model.p_grainpool_proportion = pe.Param(model.s_crops, model.s_grain_pools, initialize=params['grain_pool_proportions'], default = 0.0, doc='proportion of grain in each pool')
    
    model.p_grain_price = pe.Param(model.s_phase_periods, model.s_enterprises, model.s_season_periods, model.s_season_types, model.s_grain_pools, model.s_crops, initialize=params['grain_price'],default = 0.0, doc='farm gate price per tonne of each grain')
    
    model.p_grain_wc = pe.Param(model.s_phase_periods, model.s_enterprises, model.s_season_periods, model.s_season_types, model.s_grain_pools, model.s_crops, initialize=params['grain_wc'],default = 0.0, doc='farm gate wc per tonne of each grain')
    
    model.p_phasesow_req = pe.Param(model.s_phases, model.s_crops, model.s_lmus, initialize=params['phase_sow_req'], default = 0.0, doc='ha of sow activity required by each rot phase')
    
    model.p_sow_prov = pe.Param(model.s_phase_periods, model.s_labperiods, model.s_season_types, model.s_landuses, initialize=params['sow_prov'], default = 0.0, doc='states which landuses can be sown in each p5 period')


#######################################################################################################################################################
#######################################################################################################################################################
#functions used in global constraints
#######################################################################################################################################################
#######################################################################################################################################################

##############
#yield       #
##############
##total grain transfer for each crop. This is initially separated from cashflow so it can be combined with untimely sowing and crop grazing penalty.
### slightly more complicated because i have to have rotation yield in disaggregated format and the rotation variable is aggregated.
### yield needs to be disaggregated so that it returns the grain transfer for each crop - this is so it is compatible with yield penalty and sup feed activities.
### alternative would have been to add another key/index/set to the yield parameter that was k, although i suspect this would make it a bit slower due to being bigger but it might be tidier

def f_rotation_yield(model,m,g,k,z):
    '''
    Calculate the total (kg) of each grain harvested from selected rotation phases.

    Used in global constraint (con_grain_transfer). See CorePyomo
    '''
    return sum(model.p_rotation_yield[r,k,l,z,m]*model.v_phase_area[m,z,r,l]
               for r in model.s_phases for l in model.s_lmus
               if pe.value(model.p_rotation_yield[r,k,l,z,m]) != 0) * model.p_grainpool_proportion[k,g]



##############
#sow         #
##############
##similar to yield - this is more complex because we want to mul with phase area variable then sum based on the current landuse (k)
##returns a tuple, the boolean part indicates if the constraint needs to exist
def f_phasesow_req(model,m,k,l,z):
    '''
    Calculate the seeding requirement for each rotation phase.

    Used in global constraint (con_sow). See CorePyomo
    '''
    if any(model.p_phasesow_req[r,k,l] for r in model.s_phases):
        return sum(model.p_phasesow_req[r,k,l]*model.v_phase_increment[m,z,r,l] for r in model.s_phases
                   if pe.value(model.p_phasesow_req[r,k,l]) != 0)
    else:
        return 0


#####################################
# functions used to define cashflow #
#####################################

def f_rotation_cost(model,c0,p7,z):
    '''
    Calculate the total cost of the selected rotation phases.

    Used in objective. See CorePyomo
    '''

    return sum(model.p_rotation_cost[c0,p7,z,m,l,r]*model.v_phase_area[m,z,r,l]
               + model.p_increment_rotation_cost[c0,p7,z,l,r,m]*model.v_phase_increment[m,z,r,l]
               for r in model.s_phases for l in model.s_lmus for m in model.s_phase_periods
                   if pe.value(model.p_rotation_cost[c0,p7,z,m,l,r]) != 0)

def f_rotation_wc(model,c0,p7,z):
    '''
    Calculate the total wc of the selected rotation phases.

    Used in global constraint (con_workingcap). See CorePyomo
    '''

    return sum(model.p_rotation_wc[c0,p7,z,m,l,r]*model.v_phase_area[m,z,r,l]
               + model.p_increment_rotation_wc[c0,p7,z,l,r,m]*model.v_phase_increment[m,z,r,l]
               for r in model.s_phases for l in model.s_lmus for m in model.s_phase_periods
                   if pe.value(model.p_rotation_wc[c0,p7,z,m,l,r]) != 0)



















      ######
    # # Area #
    # ########
    # #area of rotation on a given soil can't be more than the amount on that soil available on farm
    # def area_rule(model, l):
    #   return sum(model.area_phase[r,l] for r in model.phases) <= model.area[l] 
    # model.area_con = Constraint(model.lmus, rule=area_rule, doc='rotation area constraint')
    
    # #####################
    # # lo bound rotation #
    # #####################
    # #area of rotation on a given soil can't be more than the amount on that soil available on farm
    # def rot_lo_bound(model, r1,r2,r3,r4,r5, l):
    #   return model.area_phase[r1,r2,r3,r4,r5,l] >= model.lo[r1,r2,r3,r4,r5] 
    # model.rot_bound = Constraint(model.phases, model.lmus, rule=rot_lo_bound, doc='lo bound for the number of each phase')
    
    # ##build and define rotation constraint 1 - used to ensure that the each rotation provides and requires one or more histories
    # ##alternative method (a1 - michael)
    # def rot_phase_link(model,l,h1,h2,h3,h4):
    #     return sum(model.area_phase[r,l]*model.rot_phase_link[r,h1,h2,h3,h4] for r in model.phases if ((r)+(h1,)+(h2,)+(h3,)+(h4,)) in model.rot_phase_link)<=0
    # model.rotation_con1 = Constraint(model.lmus,model.rot_constraints, rule=rot_phase_link, doc='rotation phases constraint')
    
    # # ##build and define rotation constraint 2 - used to ensure that the history provided by a rotation is used by another rotation (because one rotation can provide multiple histories)
    # # def rot_phase_link2(model,l,h1,h2,h3,h4):
    # #     return sum(model.area_phase[r,l]*model.rot_phase_link2[r,h1,h2,h3,h4] for r in model.phases if ((r)+(h1,)+(h2,)+(h3,)+(h4,)) in model.rot_phase_link2)<=0
    # # model.rotation_con2 = Constraint(model.lmus,model.rot_constraints2, rule=rot_phase_link2, doc='rotation phases constraint')


#    model.del_component(model.j)
#print(timeit.timeit(test3,number=10)/10)
#model.j.pprint()
    # return rotation_income(model,c)#- fert_cost(model,c)#- fert_app_cost(model,c) - stubble_handling_cost(model,c) #



# def rot_phase_link(model,l,h1):
#     print(h1)
#     return sum(model.area_phase[r,l]*model.rot_phase_link[r,h1] for r in model.phases if ((r)+(h1,)) in model.rot_phase_link)<=0
# model.jj = Constraint(model.lmus,model.rot_constraints, rule=rot_phase_link, doc='')

# model.jj.pprint()
#model.del_component(model.jj)
#model.del_component(model.jj_index)









#old method below:

#'''
##constrains
#'''
#########
## Area #
#########
##area of rotation on a given soil can't be more than the amount on that soil available on farm
#def area_rule(model, l):
#  return sum(model.area_phase[l,r] for r in model.phases) <= model.area[l] 
#model.area_con = Constraint(model.lmus, rule=area_rule, doc='rotation area constraint')
#
######################################
## functions used to define cashflow #
######################################
##cost of fert application - 1) per tonne 2) per ha 
#def fert_app_cost(model,c):
#    per_t_cost = sum(sum(sum(model.phasefert[r,l,n]*model.area_phase[l,r]*(model.fert_app_cost_tonne[c,n]/1000)  for r in model.phases)for l in model.lmus)for n in model.fert_type ) 
#    per_ha_cost = sum(sum(model.area_phase[l,r]*(model.fert_app_cost_ha[r,c,l]/1000) for r in model.phases) for l in model.lmus)   
#    return per_t_cost + per_ha_cost 
#
##cost of fert for rotations for a give cashflow period (c), 
#def fert_cost(model,c):
#    return sum(sum(sum(model.phasefert[r,l,n]*model.area_phase[l,r] for r in model.phases) for l in model.lmus)*(model.fert_type_cost[c,n]/1000) for n in model.fert_type ) 
#
##cost of stubble handling for each phase for a give cashflow period (c), 
#def stubble_handling_cost(model,c):
#    return sum(sum(model.stubble_handling_prob[r,l]*model.area_phase[l,r]* model.stubble_handling_cost[c] for r in model.phases)for l in model.lmus) 
#
##costs of rotations excluding fert for a give cashflow period (c)
##def rotation_cost(model,c):
##    return sum(sum(model.rotationcost[i,j,c]*model.x[i,j] for j in model.rotations)for i in model.lmus) 
##rotation income (grain income) for a give cashflow period (c)
#def rotation_income(model,c):
#    return sum(sum(model.rotationyield[r,l]*model.area_phase[l,r]*(model.grainincome[r,c]/1000) for r in model.phases)for l in model.lmus) 
#
##overall cashflow function (sums up functions above, this could all just be one but i am thinking this looks more simple)
#def rotation_cashflow(model,c):
#    return rotation_income(model,c)- fert_cost(model,c)- fert_app_cost(model,c) - stubble_handling_cost(model,c) #+ model.x[c] >=0
##model.rotation_con = Constraint(model.cashflow_periods, rule=rotation_cashflow, doc='')
#

    # #calling a function multiple times takes time. call it once and assign result to a unique variable. 
    # #local variables are easier for python to locate
    # # a=phs.phase_yields()
    # # b=phs.fert()
    # # c=phs.stubble_handling_prob()
    # # d=phs.grain_price()
    # #e=phs.fert_cost()
    # # f=mac.fert_app_t()
    # # g=mac.fert_app_ha()
    # h=phs.rot_phase_mps()
    # g=phs.rot_phase_mps2()
    # #'''
    # ##define parameters



# #cost of fert application - 1) per tonne 2) per ha 
# def fert_app_cost(model,c):
#     #per_t_cost = sum(model.phasefert[rln[:-2],rln[-2],rln[-1]]*model.area_phase[rln[:-2],rln[-2]]*(model.fert_app_cost_tonne[c,rln[-1]]/1000) for rln in model.phasefert if ((c,)+(rln[-1],)) in model.fert_type_cost) 
#     per_ha_cost = sum(model.area_phase[rlc[:-2],rlc[-2]]*(model.fert_app_cost_ha[rlc[:-2],rlc[-2],c]/1000) for rlc in model.fert_app_cost_ha if ((rlc[-1],)+(rlc[-1],)+(c,)) in model.fert_app_cost_ha) 
#     return per_t_cost + per_ha_cost 

#cost of fert for rotations for a give cashflow period (c), 
#def tesft():
    
# def fert_cost(model,c):
#     return sum(model.rotation_cost[rlc[:-2],rlc[-2],c]*model.area_phase[rlc[:-2],rlc[-2]] for rlc in model.rotation_cost  in model.rotation_cost if (rlc[:-2]+(rlc[-2],)+(c,)))+ model.x[c] >=0
# model.rotation_con = Constraint(model.cashflow_periods, rule=fert_cost, doc='')
#model.rotation_con.pprint()
#     model.del_component(model.rotation_con)
# print(timeit.timeit(tesft,number=10)/10)

# #cost of stubble handling for each phase for a give cashflow period (c), 
# def stubble_handling_cost(model,c):
#     return sum(model.stubble_handling_prob[rl[:-1],rl[-1]]*model.area_phase[rl[:-1],rl[-1]]*(model.stubble_handling_cost[c]) for rl in model.stubble_handling_prob if c in model.stubble_handling_cost) 


#rotation income (grain income) for a give cashflow period (c)
#def test2():
# def rotation_income(model,c):
#     return sum(model.rotationyield[rl[:-1],rl[-1]]*model.area_phase[rl[:-1],rl[-1]]*(model.grainincome[c,rl[-2]]/1000) for rl in model.rotationyield if ((c,)+(rl[-2],)) in model.grainincome)#+ model.x[c] >=0 #0.0677s
  #  model.rotation_con2 = Constraint(model.cashflow_periods, rule=rotation_income, doc='')
   # model.del_component(model.rotation_con2)
#print(timeit.timeit(test2,number=10)/10)
#model.rotation_cashflow.pprint()
#total costs and income from rotations - transferred to core model

# def test2():
#     def rotation_cashflow(model,c):
#         return sum(model.rotation_cashflow[rl[:-2],rl[-2],c]*model.area_phase[rl[:-2],rl[-2]] for rl in model.rotation_cashflow if (rl[:-2]+(rl[-2],)+(c,)) in model.rotation_cashflow)+ model.x[c] >=0 #0.12677s
#     model.j = Constraint(model.cashflow_periods, rule=rotation_cashflow, doc='')
#     model.del_component(model.j)
# print(timeit.timeit(test2,number=10)/10)
    

    # # model.rotation_cost= Param(phs.f.keys(), initialize=phs.f)
    # #model.rotation_cost.pprint()
    # # model.rotation_cashflow = Param(phs.rot_cashflow().keys(), initialize=phs.rot_cashflow(), doc='total cashflow for 1 unit of rotation')
    # # model.phasefert = Param(b.keys(), initialize=b, doc='fert required by 1 unit of phase')
    # # model.stubble_handling_prob = Param(c.keys(), initialize=c, doc='probability of each phase that requires stubble handling')
    # # model.stubble_handling_cost = Param(model.cashflow_periods, initialize=mac.stubble_cost_ha(),default = 0.0, doc='cost to handle 1ha of stubble')
    # model.area = Param(model.lmus, initialize=gi.general_input['lmu_area'], doc='available area on farm for each soil') #alternate way to initialise a parameter
    # model.lo = Param(model.phases, initialize=phs.lo_bound, doc='lo bound of the number of ha of rot_phase')
    
    # #model.area = Param(model.lmus, initialize=gi.general_input['lmu_area'], doc='available area on farm for each soil') #alternate way to initialise a parameter
    # # model.grainincome = Param(d.keys(), initialize=d, doc='farm gate price per tonne of each grain')
    # # model.fert_type_cost = Param(e.keys(), initialize=e, doc='price per tonne of each fert')
    # # model.fert_app_cost_tonne = Param(f.keys(), initialize= f, doc='cost of fert application per tonne of each fert (filling up and driving to paddock cost)')
    # # model.fert_app_cost_ha = Param(g.keys(), initialize= g, doc='cost of fert application per ha of each fert (driving around paddock cost)')
    # model.rot_phase_link= Param(h.keys(), initialize=h, doc='link between rotation history and current rotation')
    # model.rot_phase_link2= Param(g.keys(), initialize=g, doc='link between rotation history2 and current rotation')
    # #model.rot_phase_link.pprint() #need to fix the key for the dict so that it gets the correct set indexing in the param
    # '''
    # #define parameters, this method of defining params is 20-30% faster, doesn't need to calc the .keys method as above.
    # '''
    
    #requirement parameters for rotations are read in from csv but to complete model i have hand inputted some parameters
    #    model.rotationyield = Param(model.phases, model.lmus, initialize=phs.phase_yields(), default = 0.0, doc='grain production for all crops for 1 unit of rotation')
    #    model.phasefert = Param(model.phases, model.lmus, model.fert_type, initialize=phs.fert(), default = 0.0, doc='fert required by 1 unit of phase')
    #    model.stubble_handling_prob = Param(model.phases, model.lmus, initialize=phs.stubble_handling_prob(), default = 0.0, doc='probability of each phase that requires stubble handling')
    #    model.stubble_handling_cost = Param(model.cashflow_periods, initialize=mac.stubble_cost_ha(),default = 0.0, doc='cost to handle 1ha of stubble')
    #    model.area = Param(model.lmus, initialize=gi.general_input['lmu_area'], doc='available area on farm for each soil') #alternate way to initialise a parameter
    #    model.grainincome = Param(model.cashflow_periods, model.crops, initialize=phs.grain_price(),default = 0.0, doc='farm gate price per tonne of each grain')
    #    model.fert_type_cost = Param(model.cashflow_periods, model.fert_type, initialize=phs.fert_cost(), default = 0.0, doc='price per tonne of each fert')
    #    model.fert_app_cost_tonne = Param(model.cashflow_periods, model.fert_type, initialize= mac.fert_app_t(), default = 0.0, doc='cost of fert application per tonne of each fert (filling up and driving to paddock cost)')
    #    model.fert_app_cost_ha = Param(model.phases, model.cashflow_periods, model.lmus, initialize= mac.fert_app_ha(), default = 0.0, doc='cost of fert application per ha of each fert (driving around paddock cost)')
    #    model.stubble = Param(model.crops, initialize=phs.stubble_production(), default = 0.0, doc='stubble produced / kg grain harvested')
    