"""

This module determines the proportion of total stubble in each category based on liveweight data from paddock trials.
DMD for each category is an input along with information about the sheep in the paddock trials. Stockgenerator is
then run to determine the LWC (live weight change) provided by each category. Based of off the actual trial LWC
the proportion of stubble in each category is determined (every category is set to have at least some stubble
so that transferring can always occur).

Stockgenerator is run for all sheep groups and then the animal that reflects the trial animal is selected.
The different DMD levels are reflected along the w axis.

If there are multiple paddock trials that are being used to simulate stubble there are two options:

    1. Average the paddock live weight
    2. Simulate each trial differently and then average the results.

The choice of option would be dependent on whether there is information about the livestock for each trial. If there
is no information about the sheep in each trial then there is no benefit of simulating each trial separately.

The module writes the answers to an excel book which is referenced by AFO - this means this module only needs to be run
if you make changes to stub inputs or if you change the inputs/formulas that calculate lwc in the generator (it could be
made so that the stubble generator is run for every trial but this will slow AFO and potentially confuse the interpretation
of results).

27/2/2022: currently there is only liveweight data for GSW region. Thus CWW uses the same stubble. This would be improved
by running a stubble trial in the wheatbelt and recording liveweight every week.

@author: young
"""
#python modules
import pandas as pd
import numpy as np
from scipy.optimize import minimize

#AFO modules
import UniversalInputs as uinp
import PropertyInputs as pinp
import StructuralInputs as sinp
import StockFunctions as sfun
import Functions as fun
import Sensitivity as sen
import StockGenerator as sgen
import CropResidue as stub
from . import relativeFile

na = np.newaxis


###############
#User control #
###############
trial = 4   #4 is quick test

##sort exp
exp_data, exp_group_bool, trial_pinp = fun.f_read_exp()
exp_data = fun.f_group_exp(exp_data, exp_group_bool)
##select property for the current trial
pinp.f_select_pinp(trial_pinp.iloc[trial])

##update sensitivity values
sen.create_sa()
fun.f_update_sen(trial,exp_data,sen.sam,sen.saa,sen.sap,sen.sar,sen.sat,sen.sav)
##call sa functions - assigns sa variables to relevant inputs
sinp.f_structural_inp_sa()
uinp.f_universal_inp_sa()
pinp.f_property_inp_sa()
##expand p6 axis to include nodes
sinp.f1_expand_p6()
pinp.f1_expand_p6()

#################
#inputs for sgen#
#################
##inputs are stored in a dict
stubble_inp = {}
##create fs - read from inputs

##sim run periods - start and end p
trial_commencement_date = pinp.stubble['start_trial']
n_sim_periods, date_start_p, date_end_p, p_index_p, step \
    = sfun.f1_sim_periods(sinp.stock['i_sim_periods_year'], sinp.stock['i_age_max'])

###scale trial start to the correct yr in the sim based on animal age
add_yrs = np.ceil((date_start_p[0] - trial_commencement_date) / 364)
# sub_yrs = np.ceil(np.maximum(0, (item_start - end_of_periods).astype('timedelta64[D]').astype(int) / 365))
trial_commencement_date = trial_commencement_date + add_yrs * 364
####scale for animal age
trial_commencement_date = trial_commencement_date + pinp.stubble['animal_age'] * 364

##general info
b0_pos = sinp.stock['i_b0_pos']
b1_pos = sinp.stock['i_b1_pos']
p_pos = sinp.stock['i_p_pos']
s1_pos = sinp.stock['i_w_pos'] #s1 goes in w pos for the stubble sim

len_k = len(sinp.landuse['C'])
len_s2 = len(pinp.stubble['i_idx_s2'])
len_p1 = n_sim_periods
len_s1 = len(pinp.stubble['i_stub_cat_dmd_s1'])

##determine sgen run periods
p_start_trial = np.searchsorted(date_start_p, trial_commencement_date)
p_start_harv = np.searchsorted(date_start_p, trial_commencement_date - pinp.stubble['i_calibration_offest'])
p_end = p_start_trial + pinp.stubble['trial_length']
stubble_inp['p_start'] = p_start_trial
stubble_inp['p_end'] = p_end

##lw each period - based off fitting quadratic to the paddock data.
days_since_trialstart_p = (date_start_p - date_start_p[p_start_trial]).astype(int) #days since trial start
a_ks2 = pinp.stubble['i_a_ks2']
b_ks2 = pinp.stubble['i_b_ks2']
c_ks2 = pinp.stubble['i_c_ks2']
trial_lw_pks2 = a_ks2*days_since_trialstart_p[:,na,na] + b_ks2*days_since_trialstart_p[:,na,na]**2 + c_ks2

##dmd categories to generate - include deterioration.
## deteriortation is since harvest because the definition of the categories are at harvest.
dmd_s1 = pinp.stubble['i_stub_cat_dmd_s1'] #at harvest
days_since_harv_p = (date_start_p - date_start_p[p_start_harv]).astype(int) #days since harvest
dmd_ps1k = dmd_s1[:,na] * (1 - pinp.stubble['quality_deterioration']) ** days_since_harv_p[:,na,na]

##initilise arrays so they can be assigned by k
lwc_p1s1ks2 = np.zeros((len_p1,len_s1,len_k,len_s2))
intake_p1s1ks2 = np.zeros((len_p1,len_s1,len_k,len_s2))

for k in range(len_k):
    for s2 in range(len_s2):
        ##call stock gen
        stubble_inp['lw'] = trial_lw_pks2[:,k,s2]
        stubble_inp['dmd_pw'] = dmd_ps1k[:,:,k]
        o_pi_tpdams, o_pi_tpoffs, o_ebg_tpdams, o_ebg_tpoffs = sgen.generator(stubble=stubble_inp)

        ##slice based on animal in trial
        ## currently only the g and b axis are selected based on trial info. Any other axes are averaged. (This could be changed).
        if pinp.stubble['i_dams_in_trial']:
            ###select across g axis - weighted
            mask_dams_inc_g1 = np.any(sinp.stock['i_mask_g1g3'] * pinp.sheep['i_g3_inc'], axis=1)
            mask_offs_inc_g3 = np.any(sinp.stock['i_mask_g3g3'] * pinp.sheep['i_g3_inc'], axis=1)
            o_ebg_tpdams = np.compress(mask_dams_inc_g1, o_ebg_tpdams, axis=-1)
            o_pi_tpdams = np.compress(mask_dams_inc_g1, o_pi_tpdams, axis=-1)
            ###select across b axis - weighted
            i_b1_propn_b1g = fun.f_expand(pinp.stubble['i_b1_propn'], b1_pos)
            lwc_ps1g = np.sum(o_ebg_tpdams * i_b1_propn_b1g, b1_pos, keepdims=True)
            intake_ps1g = np.sum(o_pi_tpdams * i_b1_propn_b1g, b1_pos, keepdims=True)
            ###average remaining axes
            lwc_p1s1ks2[:,:,k,s2] = fun.f_reduce_skipfew(np.average, lwc_ps1g, preserveAxis=(p_pos, s1_pos))
            intake_p1s1ks2[:,:,k,s2] = fun.f_reduce_skipfew(np.average, intake_ps1g, preserveAxis=(p_pos, s1_pos))
        else:
            ###select across g axis - weighted
            mask_offs_inc_g3 = np.any(sinp.stock['i_mask_g3g3'] * pinp.sheep['i_g3_inc'], axis=1)
            o_ebg_tpoffs = np.compress(o_ebg_tpoffs, mask_offs_inc_g3, axis=-1)
            o_pi_tpoffs = np.compress(o_pi_tpoffs, mask_offs_inc_g3, axis=-1)
            ###select across b axis - weighted
            i_b0_propn_b0g = fun.f_expand(pinp.stubble['i_b0_propn'], b0_pos)
            lwc_ps1g = np.sum(o_ebg_tpoffs * i_b0_propn_b0g, b0_pos, keepdims=True)
            intake_ps1g = np.sum(o_pi_tpoffs * i_b0_propn_b0g, b0_pos, keepdims=True)
            ###average remaining axes
            lwc_p1s1ks2[:,:,k,s2] = fun.f_reduce_skipfew(np.average, lwc_ps1g, preserveAxis=(p_pos, s1_pos))
            intake_p1s1ks2[:,:,k,s2] = fun.f_reduce_skipfew(np.average, intake_ps1g, preserveAxis=(p_pos, s1_pos))

##post process the lwc
###calc trial lw with p1p2 axis (p2 axis is days)
len_p2 = int(step)
index_p2 = np.arange(len_p2)
date_start_p1p2 = date_start_p[..., na] + index_p2
days_since_trialstart_p1p2 = (date_start_p1p2 - date_start_p[p_start_trial,na]).astype(int)  # days since trial start
lw_p1p2ks2 = a_ks2 * days_since_trialstart_p1p2[:,:,na,na] + b_ks2 * days_since_trialstart_p1p2[:,:,na,na] ** 2 + c_ks2
lw_pks2 = lw_p1p2ks2.reshape(-1,len_k,len_s2)
trial_lwc_pks2 = np.roll(lw_pks2, shift=-1, axis=0) - lw_pks2
trial_lwc_p1p2ks2 = trial_lwc_pks2.reshape(-1,len_p2, len_k, len_s2)
###calc grazing days in generator period for each dmd - allocate trial lwc to the simulated lwc and sum the p2
lwc_diff_p1p2s1ks2 = np.abs(lwc_p1s1ks2[:,na,:,:,:] - trial_lwc_p1p2ks2[:,:,na,:,:])
grazing_days_p1s1ks2 = np.sum(np.equal(np.min(lwc_diff_p1p2s1ks2, axis=2,keepdims=True) , lwc_diff_p1p2s1ks2), axis=1)
###adjust intake - allowing for decay related to quantity (to reflect the amount at harvest). (Trampling done below).
adj_intake_p1s1ks2 = intake_p1s1ks2 / (1 - pinp.stubble['quantity_decay'][:,na]) ** days_since_harv_p[:, na, na, na]
###multiply by adjusted intake and sum p axis to return the total intake for each dmd (stubble) category
total_intake_s1ks2 = np.sum(grazing_days_p1s1ks2 * adj_intake_p1s1ks2, axis=0)
total_intake_ha_s1ks2 = total_intake_s1ks2 * pinp.stubble['i_sr_s2']
###adjust for trampling - trampling is done as a percentage of consumed stubble thus trampling doesnt remove categories above because they have already been consumed.
### Trampling gets added on to reflect the amount of stubble at harvest.
tramp_ks2 = pinp.stubble['trampling'][:,na]
total_intake_ha_s1ks2 = total_intake_ha_s1ks2 + tramp_ks2 * np.cumsum(total_intake_ha_s1ks2, axis=0)
###set a minimum for each category so that the transfer between cats can always occur.
total_intake_ha_s1ks2 = np.maximum(1, total_intake_ha_s1ks2) #minimum of 1kg in each category so stubble can always be transferred between categories.
###divide intake by total stubble to return stubble proportion in each category
harvest_index_k = pinp.stubble['i_harvest_index_ks2'][:,0] #select the harvest s2 slice because yield penalty is inputted as a harvestable grain
biomass_k = pinp.stubble['i_trial_yield'] / harvest_index_k
total_residue_ks2 = biomass_k[:,na] * stub.f_biomass2residue(residuesim=True)
cat_propn_s1ks2 = total_intake_ha_s1ks2/total_residue_ks2

# Create a Pandas Excel writer using XlsxWriter as the engine. used to write to multiple sheets in excel
stubble_sim_path = relativeFile.findExcel('stubble sim.xlsx')
writer = pd.ExcelWriter(stubble_sim_path, engine='xlsxwriter')
cat_propn_s1_ks2 = pd.DataFrame(cat_propn_s1ks2.reshape(len_s1,len_k*len_s2))
cat_propn_s1_ks2.to_excel(writer,index=False,header=False)
writer.save()



#########
#old sim#
#########
# ##inputs
# hi_k = pinp.stubble['harvest_index']
# index_k = pinp.stubble['i_stub_landuse_idx']
# proportion_grain_harv_k = pinp.stubble['proportion_grain_harv']
# stub_cat_prop_ks1 = pinp.stubble['stub_cat_prop']
#
# ##calc the dmd of each component at the point when category dmd was calibrated
# deterioration_factor_ks0 = pinp.stubble['quality_deterioration']
# days_since_harv = pinp.stubble['i_calibration_offest']
# dmd_component_harv_ks0 = pinp.stubble['component_dmd'] #dmd at harvest
# dmd_component_ks0 = ((1 - deterioration_factor_ks0) ** days_since_harv) * dmd_component_harv_ks0
#
#
# for crp in range(len(index_k)):
#     ######
#     #sim #
#     ######
#
#     def grain_prop():
#         '''calc grain propn in stubble
#
#         HI = total grain / total biomass (total biomass includes grain as well)
#         stubble = leaf and stalk plus split grain
#         '''
#         hi = hi_k[crp]
#         harv_prop = proportion_grain_harv_k[crp]
#         splitgrain_propn_totalbiomass = hi*(1-harv_prop) #split grain as a propn of total biomass
#         leafstalk_propn_totalbiomass = (1-hi) #leaf & stalk as a propn of total biomass
#         stubble_propn_totalbiomass = splitgrain_propn_totalbiomass + leafstalk_propn_totalbiomass #stubble as a propn of total biomass
#         return splitgrain_propn_totalbiomass/stubble_propn_totalbiomass * 100 #split grain as propn of stubble
#
#     #quantity of each stubble component at harvest
#     def stubble_sim(x):
#         #variables to be solved for (this is new to this version of the sim)
#         z,w,q, g, b, s, c = x
#
#
#         component_proportion={'grain' : grain_prop()
#         ,'blade' : z
#         ,'sheath': w
#         ,'chaff' : q}
#
#         #might be worth making this a proper constrain, either make stem a variable then then con is the sum of all == 100
#         component_proportion['stem'] = 100- (z+w+q+grain_prop())
#
#
#         #variables to be solved for (this is new to this version of the sim)
#         grazing_pref_component={'grain' :  g
#         ,'blade' : b
#         ,'sheath': s
#         ,'chaff' : c
#         ,'stem' :1}
#         #sim length
#         sim_length = int(100/pinp.stubble['step_size'])
#         #number of components
#         number_of_components = len(component_proportion)
#         #numpy array for each stubble section used in sim
#         stubble_availability=np.zeros([number_of_components,sim_length])
#         weighted_availability=np.zeros([number_of_components+1,sim_length]) #extra one for a total tally which is required
#         consumption=np.zeros([number_of_components,sim_length])
#         cumulative_consumption=np.zeros([number_of_components,sim_length])
#         #fill in each numpy array one step at a time. have to fill in each step for each array one at a time because the arrays are linked therefore each array used values from another
#         for step in range(sim_length):
#             #stubble availability (at the start of the sim this is component propn it then decreases depending on which components are consumed)
#             for component, proportion,component_num in zip(component_proportion.keys(),component_proportion.values(),range(number_of_components)):
#                 if step == 0:
#                     stubble_availability[component_num, step]=proportion
#                 elif stubble_availability[component_num, step-1] - consumption[component_num, step-1]<=0:
#                     stubble_availability[component_num, step]=0
#                 else: stubble_availability[component_num, step]=stubble_availability[component_num, step-1] - consumption[component_num, step-1]
#             #weighted availability (weight by consumption preference)
#             for component, proportion,component_num in zip(component_proportion.keys(),component_proportion.values(),range(len(component_proportion))):
#                 weighted_availability[component_num, step] = stubble_availability[component_num,step] * grazing_pref_component[component]
#             weighted_availability[5, step] = weighted_availability[:,step].sum()
#             #consumption per time step (consumption of each component %)
#             for component, proportion,component_num in zip(component_proportion.keys(),component_proportion.values(),range(len(component_proportion))):
#                 if weighted_availability[number_of_components,step] <= 0:
#                     consumption[component_num, step] = 0
#                 else:
#                     consumption[component_num, step] = (pinp.stubble['step_size']
#                     / weighted_availability[number_of_components,step] * weighted_availability[component_num, step] )
#             #cumulative comsumption
#             for component, proportion,component_num in zip(component_proportion.keys(),component_proportion.values(),range(len(component_proportion))):
#                 cumulative_consumption[component_num, step]= consumption[component_num].sum()
#
#         #determine the proportion of each component in each category
#         num_stub_cat = stub_cat_prop_ks1.shape[1]
#         categ_sizes = stub_cat_prop_ks1[crp,:]
#         cumulative_cat_size=[]
#         for i,j in zip(categ_sizes,range(num_stub_cat)):
#             if j > 0:
#                 cumulative_cat_size.append(cumulative_cat_size[-1]+i)
#             else: cumulative_cat_size.append(i)
#         #create numpy to store stubble dets that go into the rest of the stubble calcs
#         stub_cat_component_proportion = np.zeros([number_of_components,num_stub_cat])
#         for cat_num, cum_cat_size, cat_size in zip(range(num_stub_cat), cumulative_cat_size, categ_sizes):
#             for component in range(number_of_components):
#                 #ammount of a component consumed in a given category
#                 if cat_num == 0: #if not cat A then need to subtract off the consumed amount in the periods before
#                     comp_consumed = cumulative_consumption[component,round(cum_cat_size*100-1)]  #multiplied by 100 to convert the percent to int. it is then use to index the steps in the numpy arrays above, minus 1 because indexing starts from 0
#                 else: comp_consumed = (cumulative_consumption[component,round(cum_cat_size*100-1)] #use the cat list so that i can determine the the consumption of a component at in the cat before
#                     - cumulative_consumption[component,round(list(cumulative_cat_size)[cat_num-1]*100-1)])
#                 stub_cat_component_proportion[component, cat_num] = comp_consumed/cat_size/100
#         return stub_cat_component_proportion
#
#     def objective(x):
#         #multiplies the component dmd by the proportion of that component consumed in each cat
#         #this determines the overall dmd of that cat.
#         #the objective func minimised the diff between the value above and the inputted value of cat dmd
#         # component_dmd = np.array(component_dmd, dtype=float)
#         cat_a_component_propn=stubble_sim(x)[:,0]
#         a=np.dot(cat_a_component_propn,dmd_component_ks0[crp,:])
#         cat_b_component_propn=stubble_sim(x)[:,1]
#         b=np.dot(cat_b_component_propn,dmd_component_ks0[crp,:])
#         cat_c_component_propn=stubble_sim(x)[:,2]
#         c=np.dot(cat_c_component_propn,dmd_component_ks0[crp,:])
#         cat_d_component_propn=stubble_sim(x)[:,3]
#         d=np.dot(cat_d_component_propn,dmd_component_ks0[crp,:])
#         cat_a_target = pinp.stubble['stub_cat_qual'][crp,0]
#         cat_b_target = pinp.stubble['stub_cat_qual'][crp,1]
#         cat_c_target = pinp.stubble['stub_cat_qual'][crp,2]
#         cat_d_target = pinp.stubble['stub_cat_qual'][crp,3]
#
#         return ((a-cat_a_target)**2+(b-cat_b_target)**2+(c-cat_c_target)**2+(d-cat_d_target)**2)
#     #initial guesses
#     x0 = np.ones(7)
#     # bounds on variables
#     bndspositive = (0, 100.0) #qualtity of other components must be greater than 10%
#     no_upbnds = (1, 1.0e10) #pref has to be greater than stem
#     if index_k[crp] in ('r', 'z', 'l', 'f'):   #because these crops only have 4 stubble components ie no sheath
#         var_bound = (0,10) #still need to give optimisation some room to move otherwise it gives bad solution.
#     else: var_bound = (0,100)
#     bnds = (bndspositive, var_bound, bndspositive, no_upbnds, no_upbnds, no_upbnds, no_upbnds)
#     #may have to change around the solver (method) to get the best solution
#     solution = minimize(objective, x0, method='SLSQP', bounds=bnds)
#     x = solution.x
#     stub_cat_component_proportion = pd.DataFrame(stubble_sim(x))
#     stub_cat_component_proportion.to_excel(writer, sheet_name=index_k[crp],index=False,header=False)
#
#     #################################################
#     #post calcs to make sure everything looks good  #
#     #################################################
#     #check the component proportion
#     component_proportion={'grain' : grain_prop()
#         ,'blade' : x[0]
#         ,'sheath': x[1]
#         ,'chaff' : x[2]
#         ,'stem': 100- (x[0]+x[1]+x[2]+grain_prop())}
#     grazing_pref_component={'grain' :  x[3]
#         ,'blade' : x[4]
#         ,'sheath': x[5]
#         ,'chaff' : x[6]
#         ,'stem' :1}
#
#     def cat_ddm(x):
#         #multiplies the component dmd by the proportion of that component consumed in each cat
#         #this determines the overall dmd of that cat.
#         #the objective func minimised the diff between the value above and the inputted value of cat dmd
#         cat_a_dmd=stubble_sim(x)[:,0]
#         a=np.dot(cat_a_dmd,dmd_component_ks0[crp,:])
#         cat_b_dmd=stubble_sim(x)[:,1]
#         b=np.dot(cat_b_dmd,dmd_component_ks0[crp,:])
#         cat_c_dmd=stubble_sim(x)[:,2]
#         c=np.dot(cat_c_dmd,dmd_component_ks0[crp,:])
#         cat_d_dmd=stubble_sim(x)[:,3]
#         d=np.dot(cat_d_dmd,dmd_component_ks0[crp,:])
#         return(a,b,c,d)
#
#     print('-'*100)
#     print(index_k[crp])
#     print('component proportions at harv : ',component_proportion.values()) #dict values, check to make sure they look sensible
#     print('graxing pref : ',grazing_pref_component.values()) #dict values, check to make sure they look sensible
#     print('cat ddm : ',cat_ddm(x))
#     print('Target cat ddm : ',pinp.stubble['stub_cat_qual'][crp,0],pinp.stubble['stub_cat_qual'][crp,1], pinp.stubble['stub_cat_qual'][crp,2],pinp.stubble['stub_cat_qual'][crp,3])
#     print('objective : ',objective(x))
#
# writer.save()
#
#