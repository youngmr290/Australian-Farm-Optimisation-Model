'''

Trees are represented differently in AFO than other land uses. The area of trees on each LMU is a user input. 
This means the area of trees can not be optimised. Optimising the area of trees is complicated because of the
interactions between trees and adjacent paddocks. This means trees can't be represented as a single variable 
because increasing the area of trees needs to be linked to the production of other landuses on the LMU.
To optimise the area of trees would require duplicating all the land uses. 
For example there would need to be a wheat land use and a wheat with trees land use.



'''




import numpy_financial as npf
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from . import UniversalInputs as uinp
from . import PropertyInputs as pinp
from . import StructuralInputs as sinp
from . import Finance as fin
from . import Functions as fun
from . import SeasonalFunctions as zfun
from . import Periods as per
from . import EmissionFunctions as efun
from . import Sensitivity as sen


na=np.newaxis


def f_costs(r_vals, mask_season_p7z):
    '''
    Calculate the costs of establishment and maintenance per hectare. 

    Note, the opportunity cost of the land that is affected by the trees is considered in the other functions.
    Note 2, labour is a cost rather than being hooked up to the labour constraints because it is only a once off thing (and may be completed by contractors).
    '''
    #TODO maybe need to add a fixed cost for learning etc. It would be less if going through a broker.

    # Retrieve selected plantation structure
    config = uinp.tree[f"plantation_structure_{uinp.tree['controls']['plantation_structure']}"]
    include_harvesting = uinp.tree["controls"]["include_harvesting"]
    
    # Calculate the effective width (total area used by tree rows, including side buffers)
    effective_width = (config["number_of_rows"] - 1) * config["between_row_spacing"] + 2 * config["side_buffer"]

    # Calculate planting density (plants per hectare)
    planting_density = 10000 / effective_width / config["within_row_spacing"] * config["number_of_rows"]  # trees per hectare
    # add shrubs (assume same price as trees)
    planting_density = planting_density + config["shrub_density"]  # plants per hectare
    # adjust for mortality - dead trees are replanted (or just plant at slightly higher rate to allow for mortality)
    planting_density = planting_density / (1 - uinp.tree["initial_costs"]["planting"]["mortality"])

    # Get LMU data as NumPy arrays
    lmu_fert_scalar_l = pinp.tree["tree_fert_soil_scalar"]

    # Establishment costs (Year 1)
    initial_costs = uinp.tree["initial_costs"]
    establishment_costs_l = (
        initial_costs["site_prep"]["ripping_mounding"] +
        initial_costs["site_prep"]["initial_weed_control"] +
        (initial_costs["site_prep"]["fertiliser_base"] +
        initial_costs["site_prep"]["fertiliser_extra_if_harvested"] * include_harvesting) * lmu_fert_scalar_l +
        initial_costs["planting"]["seedlings"] * planting_density +
        (planting_density / initial_costs["planting"]["planting_rate"]) * (initial_costs["planting"]["farmer_labour_planting"] + initial_costs["planting"]["planting_equipment"])
    )
    
    # Maintenance costs (Year 2 and beyond)
    yr_1 = uinp.tree["yr_1_costs"]
    yr1_costs_l = yr_1["weed_control"] + (yr_1["fertiliser_base"] + yr_1["fertiliser_extra_if_harvested"] * include_harvesting) * lmu_fert_scalar_l
    
    # Long-term maintenance (Years 3-100)
    yrs_2_to_100 = uinp.tree["yrs_2_to_100"]
    yrs_2_to_100_costs_l = yrs_2_to_100["weed_control"] + (yrs_2_to_100["fertiliser_base"] + yrs_2_to_100["fertiliser_extra_if_harvested"] * include_harvesting) * lmu_fert_scalar_l
    
    # Discounted present value of future maintenance costs
    discount_rate  = uinp.finance['i_interest']
    num_years = uinp.tree["controls"]["project_duration"]  # Total number of years

    # Construct cash flow array
    # Build a list of arrays first
    cost_rows = [establishment_costs_l, yr1_costs_l,] + [yrs_2_to_100_costs_l] * (num_years - 1)
    # Stack them into a 2D array (shape: [num_years, n])
    total_cash_flows_yl = np.vstack(cost_rows)

    # Compute NPV 
    npv_l = fun.f_npv_nd(total_cash_flows_yl, discount_rate)
    
    # Compute annualized cost as an annuity
    annual_payment_l = npv_l * (discount_rate / (1 - (1 + discount_rate) ** -num_years))
    
    #allocate to p7
    cost_allocation_p7z, wc_allocation_c0p7z = fin.f_cashflow_allocation(np.array([uinp.tree["cost_date"]]), z_pos=-1)
    tree_estab_cost_p7zl = annual_payment_l * cost_allocation_p7z[...,na]
    tree_estab_wc_c0p7zl = annual_payment_l * wc_allocation_c0p7z[...,na]
    
    ##store r_vals
    fun.f1_make_r_val(r_vals, tree_estab_cost_p7zl, 'tree_estab_cost_p7zl', mask_season_p7z[...,na], z_pos=-2)
    
    return tree_estab_cost_p7zl, tree_estab_wc_c0p7zl
    
    
def f_adjacent_land_production_scalar():
    '''
    This function calculates an average yield scalar based on the costs (resources competition)
    and benefits (wind protection) of tree plantations.

    The function calculates the average relative production between belts by evaluating the
    indefinite integral of the logistic yield function between belts. The definite integral (area under the curve)
    between belts is then calculated to determine the average relative production between belts.
    
    Only one side of the tree belt receives the benefits
    of wind protection while both sides receive the resource costs.

    The logic used in the function is based on the assumption that the trees are evenly distributed across the LMU and 
    all the costs/benefits of the trees are realised in the LMU where the trees are planted.
    Meaning the costs and benefits are equally distributed over all the rotations on the LMU. 
    This assumption would be incorrect if for example the trees were in a permanant pasture paddock and other parts of the LMU were cropped.
    
    We are also assuming that the belts are far enough apart that the resource competition on the right of one belt is not felt on the left of the next belt.
    Furthermore, the current representation assumes trees have the same relative production impact on crop and pasture.
    '''
    # Get plantation configuration based on the plantation structure control setting
    plantation_structure = uinp.tree["controls"]["plantation_structure"]
    plantation_config = uinp.tree[f"plantation_structure_{plantation_structure}"]

    # Extract configuration values
    distance_between_belts = plantation_config["distance_between_belts"]  # meters between tree belts
    number_of_belts = plantation_config["number_of_belts"]
    propn_trees_adjacent = plantation_config["propn_paddock_adj"]  # Proportion of trees adjacent to usable paddock

    # Get area of trees (in hectares) as a numpy array - use estimated area unless tree area bnd is active.
    est_tree_area_l = fun.f_sa(pinp.tree["estimated_area_trees_l"], sen.sav['bnd_tree_area_l'][pinp.lmu_mask], 5)
    # Calculate the effective width of the tree rows (including side buffers)
    effective_width = ((plantation_config["number_of_rows"] - 1) * plantation_config["between_row_spacing"] +
                    2 * plantation_config["side_buffer"])

    # Calculate the length of each belt (in meters)
    belt_length_l = (est_tree_area_l * 10000) / (effective_width * number_of_belts) # Convert tree area from hectares to m² (1 ha = 10,000 m²)

    # Calculate LMU area available for production (excluding area with trees)
    lmu_area_without_trees_l = pinp.general['i_lmu_area'] - est_tree_area_l

    # Calculate the land area on the protected side of the tree belts
    protected_side_area_l = (belt_length_l * number_of_belts * distance_between_belts * propn_trees_adjacent) / 10000 # Convert m² to hectares

    # Calculate the fraction of the available LMU area (excluding tree area) that is on the protected side
    propn_lmu_protected_l = np.minimum(1, protected_side_area_l  / lmu_area_without_trees_l) # Ensure the fraction is not greater than 1 (cant shelter more than 100% of lmu)   

    # Assuming symmetry, the fraction on the non-protected side of trees is the same
    propn_lmu_nonprotected_l = propn_lmu_protected_l

    #########################################
    # Production on the PROTECTED side
    #########################################

    # parameters for the protected side (benefiting from wind protection and incurring resource competition)
    protected_side_params = plantation_config["protected_side_production_logistic_params"]
    offset_fit_protected = protected_side_params["offset"]
    k_fit_protected = protected_side_params["k"]
    x0_fit_protected = protected_side_params["x0"]
    a_fit_protected = protected_side_params["a"]
    mu_fit_protected = protected_side_params["mu"]
    sigma_fit_protected = protected_side_params["sigma"]

    # Evaluate the indefinite logistic integral at 0 and at the distance between belts
    x_max = distance_between_belts  # max distance you’re integrating to
    F0_protected = fun.f_combined_integral(0, offset_fit_protected, k_fit_protected, x0_fit_protected, a_fit_protected, mu_fit_protected, sigma_fit_protected, x_max)
    Fx_protected = fun.f_combined_integral(distance_between_belts, offset_fit_protected, k_fit_protected, x0_fit_protected, a_fit_protected, mu_fit_protected, sigma_fit_protected, x_max)

    # Compute the relative production adjustment over the belt width
    relative_production_adj_protected = (Fx_protected - F0_protected) / distance_between_belts

    # Average relative production across LMU due to resource competition on the protected side of belts
    relative_production_average_protected_l = (propn_lmu_protected_l * relative_production_adj_protected +
                                            (1 - propn_lmu_protected_l) * 1)

    #########################################
    # Production on the NON-PROTECTED side
    #########################################
    #note - assuming the belts are far enough apart that the resource competition is not felt on the other side of the tree belt.

    # Logistic parameters for the non-protected side (receiving only resource competition costs)
    nonprotected_side_params = plantation_config["nonprotected_side_production_logistic_params"]
    L_fit_nonprotected = nonprotected_side_params["L"]
    k_fit_nonprotected = nonprotected_side_params["k"]
    x0_fit_nonprotected = nonprotected_side_params["x0"]
    offset_fit_nonprotected = nonprotected_side_params["offset"]

    # Evaluate the indefinite logistic integral for non-protected side
    F0_nonprotected = fun.f_logistic_integral(0, L_fit_nonprotected, k_fit_nonprotected, x0_fit_nonprotected, offset_fit_nonprotected)
    Fx_nonprotected = fun.f_logistic_integral(distance_between_belts, L_fit_nonprotected, k_fit_nonprotected, x0_fit_nonprotected, offset_fit_nonprotected)

    # Compute the relative production adjustment over the belt width for the non-protected side
    relative_production_adj_nonprotected = (Fx_nonprotected - F0_nonprotected) / distance_between_belts

    # Average relative production across LMU due to resource competition on the non-protected side of belts
    relative_production_average_nonprotected_l = (propn_lmu_nonprotected_l * relative_production_adj_nonprotected +
                                                (1 - propn_lmu_nonprotected_l) * 1)
    
    # overall average production scalar
    average_production_scalar_l = relative_production_average_protected_l * relative_production_average_nonprotected_l
    if uinp.tree["controls"]["include_adjacent_pad_interaction"]==0:
        average_production_scalar_l[...] = 1

    return average_production_scalar_l



def f_microclimate_adj():
    '''
    This function accounts for microclimate effects of trees. The function
    calculates a windspeed scalar in paddocks adjacent to trees. The scalar is used to adjust the calculation of livestock chill.

    This assumes that livestock do not seek shelter. 
    Assumes that benefits of shelter only exist in the paddock adjacent to trees

    Chill is a non-linear function (and WS by distance from trees function). 
    Therefore, an improvement would be to report the microclimate adjusters for different parts of the paddock.

    '''
    if uinp.tree["controls"]["include_livestock_shelter"]==0:
        return 1, 1, 1
    else:
        # Get plantation configuration based on the plantation structure control setting
        plantation_structure = uinp.tree["controls"]["plantation_structure"]
        plantation_config = uinp.tree[f"plantation_structure_{plantation_structure}"]

        # Extract configuration values
        distance_between_belts = plantation_config["distance_between_belts"]  # meters between tree belts or to the edge of the paddock
        number_of_belts = plantation_config["number_of_belts"]
        propn_trees_adjacent = plantation_config["propn_paddock_adj"]  # Proportion of trees adjacent to usable paddock

        # Get total area of trees - don't need to differentiate between LMUs because ws impacts are same for all lmu
        est_tree_area_l = fun.f_sa(pinp.tree["estimated_area_trees_l"], sen.sav['bnd_tree_area_l'][pinp.lmu_mask], 5)
        est_tree_area = np.sum(est_tree_area_l)

        # Calculate the effective width of the tree rows (including side buffers)
        effective_width = ((plantation_config["number_of_rows"] - 1) * plantation_config["between_row_spacing"] +
                        2 * plantation_config["side_buffer"])

        # Calculate the length of each belt (in meters)
        belt_length = (est_tree_area * 10000) / (effective_width * number_of_belts) # Convert tree area from hectares to m² (1 ha = 10,000 m²)

        # Calculate the land area on the protected side of the tree belts
        protected_area = (belt_length * number_of_belts * distance_between_belts * propn_trees_adjacent) / 10000 # Convert m² to hectares

        #########################################
        # WS adjuster on the PROTECTED side
        #########################################

        # Logistic parameters for the protected side (benefiting from wind protection and incurring resource competition)
        ws_logistic_params = plantation_config["ws_logistic_params"]
        L_fit_ws = ws_logistic_params["L"]
        k_fit_ws = ws_logistic_params["k"]
        x0_fit_ws = ws_logistic_params["x0"]
        offset_fit_ws = ws_logistic_params["offset"]

        # Evaluate the indefinite logistic integral at 0 and at the distance between belts
        F0_ws = fun.f_logistic_integral(0, L_fit_ws, k_fit_ws, x0_fit_ws, offset_fit_ws)
        Fx_ws = fun.f_logistic_integral(distance_between_belts, L_fit_ws, k_fit_ws, x0_fit_ws, offset_fit_ws)

        # Compute the relative production adjustment over the belt width
        relative_ws_adj = (Fx_ws - F0_ws) / distance_between_belts


        #########################################
        # Temp adjuster on the PROTECTED side
        #########################################
        #todo this is not hooked up yet - need data to parameterise the logistic function.

        # Logistic parameters for the protected side (benefiting from wind protection and incurring resource competition)
        temp_logistic_params = plantation_config["temp_logistic_params"]
        L_fit_temp = temp_logistic_params["L"]
        k_fit_temp = temp_logistic_params["k"]
        x0_fit_temp = temp_logistic_params["x0"]
        offset_fit_temp = temp_logistic_params["offset"]

        # Evaluate the indefinite logistic integral at 0 and at the distance between belts
        # F0_temp = fun.f_logistic_integral(0, L_fit_temp, k_fit_temp, x0_fit_temp, offset_fit_temp)
        # Fx_temp = fun.f_logistic_integral(distance_between_belts, L_fit_temp, k_fit_temp, x0_fit_temp, offset_fit_temp)

        # Compute the relative production adjustment over the belt width
        relative_temp_adj = 1 #(Fx_temp - F0_temp) / distance_between_belts

        return relative_ws_adj, relative_temp_adj, protected_area



def f_harvestable_biomass(r_vals, mask_season_p7z):
    '''
    Calculate the income per hectare generated from harvesting tree biomass for bio-fuel. The main costs are:

        - Costs for harvest and on-farm transport
        - Road transport to the bio-energy plant
    '''
    project_duration = uinp.tree["controls"]["project_duration"]  # Total number of years
    harvested = uinp.tree["controls"]["include_harvesting"]
    biomass_harvesting = uinp.tree["biomass_harvesting"]
    biomass_price = uinp.tree["biomass_price"]
    plantation_structure = uinp.tree["controls"]["plantation_structure"]
    plantation_config = uinp.tree[f"plantation_structure_{plantation_structure}"]

    #biomass income and costs per ha
    if harvested:
        # income
        biomass_harvested_y = plantation_config["biomass_harvested_y"][0:project_duration+1] #project_duration + 1 because the first slice is t0, slice 1 is really yr 1.
        regional_growth_scalar = pinp.tree["regional_growth_scalar"]
        lmu_growth_scalar_l = pinp.tree["lmu_growth_scalar_l"]
        biomass_harvested_yl = biomass_harvested_y[:,na] * regional_growth_scalar * lmu_growth_scalar_l
        biomass_income_yl = biomass_harvested_yl * biomass_price
        # costs per tonne
        harv_cost = biomass_harvesting["contract_costs"] / biomass_harvesting["harvest_rate"] #$/t
        transport_cost = biomass_harvesting["transport_cost"] * biomass_harvesting["transport_distance"] #$/t
        # costs per hectare
        costs_yl = biomass_harvested_yl * (harv_cost + transport_cost)
    else:
        biomass_income_yl = np.zeros([project_duration+1, 1]) #project_duration + 1 because the first slice is t0, slice 1 is really yr 1.
        costs_yl = np.zeros([project_duration+1, 1]) #project_duration + 1 because the first slice is t0, slice 1 is really yr 1.
        
    # Compute NPV 
    discount_rate  = uinp.finance['i_interest']
    npv_income_l = fun.f_npv_nd(biomass_income_yl, discount_rate)
    npv_cost_l = fun.f_npv_nd(costs_yl, discount_rate)
    
    # Compute annualized cost as an annuity
    annual_income_l = npv_income_l * (discount_rate / (1 - (1 + discount_rate) ** -project_duration))
    annual_cost_l = npv_cost_l * (discount_rate / (1 - (1 + discount_rate) ** -project_duration))
    annual_cashflow_l = annual_income_l - annual_cost_l
    
    #allocate to p7
    cash_allocation_p7z, wc_allocation_c0p7z = fin.f_cashflow_allocation(np.array([uinp.tree["cost_date"]]), z_pos=-1)
    tree_biomass_income_p7zl = annual_income_l * cash_allocation_p7z[...,na]
    tree_biomass_cost_p7zl = annual_cost_l * cash_allocation_p7z[...,na]
    tree_biomass_cashflow_p7zl = annual_cashflow_l * cash_allocation_p7z[...,na]
    tree_biomass_wc_c0p7zl = annual_cashflow_l * wc_allocation_c0p7z[...,na]
    
    ##store r_vals
    fun.f1_make_r_val(r_vals, tree_biomass_income_p7zl, 'tree_biomass_income_p7zl', mask_season_p7z[...,na], z_pos=-1)
    fun.f1_make_r_val(r_vals, tree_biomass_cost_p7zl, 'tree_biomass_cost_p7zl', mask_season_p7z[...,na], z_pos=-1)
  
    return tree_biomass_cashflow_p7zl, tree_biomass_wc_c0p7zl
  
  
  


def f_sequestration(r_vals, mask_season_p7z):
    '''
    Calculates the annualised carbon sequestration cashflow and average biophysical sequestration for 1 hectare of tree plantings over a project duration,
    accounting for growth rates, fuel-related emissions, and carbon credit price.

    The function handles non-linear carbon sequestration by:
    - Estimating net CO₂e sequestered each year after accounting for fuel-related emissions.
    - Converting each year's net sequestration into a dollar value using a real carbon price.
    - Discounting these cashflows over the project duration and annualising them to align with steady-state economic conditions.
    - Separately calculating the average annual net CO₂e sequestered (in kg/ha/year) for biophysical reporting in AFO.

    This distinction ensures the financial benefit is derived from time-weighted cashflows, while the biophysical metric reflects
    the average sequestration rate. These are **not interchangeable**, since sequestration is not constant over time.
    
    Notes:
    - Sequestration is estimated using values derived from FullCAM, scaled by regional and land unit multipliers.
    - Under the Australian Carbon Credit Unit (ACCU) scheme, fuel use that generates greenhouse gas (GHG) emissions to maintain or manage vegetation does reduce the net amount of carbon sequestered and therefore reduces the number of ACCUs issued.
    - A risk of reversal buffer is applied to account for permanence uncertainty (e.g., fire, drought).
    - Income is calculated only if carbon credits are included.
    - This is assuming that the financial values in AFO are in real terms & will remain constant. In other words the opportunity cost of forgone
      agricultural production will be the same every yr in real terms. Hence the carbon price is also specfied in real terms.
    '''

    
    project_duration = uinp.tree["controls"]["project_duration"]  # Total number of years
    include_carbon_credit = uinp.tree["controls"]["include_carbon_credit"]
    carbon_price = uinp.tree["carbon_price"]
    sequestration_costs = uinp.tree["sequestration_costs"]
    plantation_structure = uinp.tree["controls"]["plantation_structure"]
    plantation_config = uinp.tree[f"plantation_structure_{plantation_structure}"]
    
    ##calc net annual sequestration per hectare
    regional_growth_scalar = pinp.tree["regional_growth_scalar"]
    lmu_carbon_scalar_l = pinp.tree["lmu_carbon_scalar_l"]
    risk_of_reversal_buffer = uinp.tree["risk_of_reversal_buffer"]
    annual_sequestration_yl = (plantation_config["annual_sequestration"][0:project_duration+1,na] * (1-risk_of_reversal_buffer)
                               * regional_growth_scalar * lmu_carbon_scalar_l) # project_duration + 1 because the first slice is t0, slice 1 is really yr 1.
    
    fuel_used_initial = uinp.tree["fuel_used"]["initial"]
    fuel_used_yr1 = uinp.tree["fuel_used"]["yr1"]
    fuel_used_yr2_to_100 = uinp.tree["fuel_used"]["yr2_to_100"]
    fuel_used_y = [fuel_used_initial, fuel_used_yr1] + [fuel_used_yr2_to_100] * (project_duration - 1)
    co2_fuel_co2e_y, ch4_fuel_co2e_y, n2o_fuel_co2e_y = efun.f_fuel_emissions(np.array(fuel_used_y))
    co2e_fuel_y = co2_fuel_co2e_y + ch4_fuel_co2e_y + n2o_fuel_co2e_y

    net_co2e_yl = annual_sequestration_yl - co2e_fuel_y[:,na]
    
    ##calc average sequestration per year for GHG report
    annual_sequestration_l = np.mean(annual_sequestration_yl, axis=0)
    co2e_fuel = np.mean(co2e_fuel_y[1:], axis=0) #dont include slice 0 in the average since slice 0 is not a real year. It just exists to account for initial costs etc.
    co2e_sold_l = (annual_sequestration_l - co2e_fuel) * include_carbon_credit
    
    #sequestration income and costs per ha
    variable_costs_y = np.zeros(project_duration+1)
    fixed_costs_y = np.zeros(project_duration+1)
    if include_carbon_credit:
        # income
        sequestration_income_yl = net_co2e_yl/1000 * carbon_price
        # costs per hectare
        variable_costs_y[1:] = sequestration_costs["annual_monitoring"]
        # costs per project
        fixed_costs_y[0] = sequestration_costs["setup"]

    else:
        sequestration_income_yl = net_co2e_yl * 0
        variable_costs_y = np.zeros([project_duration+1])
        fixed_costs_y = np.zeros([project_duration+1])

    # Compute NPV 
    discount_rate  = uinp.finance['i_interest']
    npv_income_l = fun.f_npv_nd(sequestration_income_yl, discount_rate)
    npv_variable_cost = fun.f_npv_nd(variable_costs_y, discount_rate)
    npv_fixed_cost = fun.f_npv_nd(fixed_costs_y, discount_rate)

    # Compute annualized cost as an annuity
    annual_income_l = npv_income_l * (discount_rate / (1 - (1 + discount_rate) ** -project_duration))
    annual_variable_cost = npv_variable_cost * (discount_rate / (1 - (1 + discount_rate) ** -project_duration))
    annual_fixed_cost = npv_fixed_cost * (discount_rate / (1 - (1 + discount_rate) ** -project_duration))
    annual_cashflow_l = annual_income_l - annual_variable_cost
    
    #allocate to p7
    cash_allocation_p7z, wc_allocation_c0p7z = fin.f_cashflow_allocation(np.array([uinp.tree["cost_date"]]), z_pos=-1)
    tree_sequestration_income_p7zl = annual_income_l * cash_allocation_p7z[...,na]
    tree_sequestration_variable_cost_p7z = annual_variable_cost * cash_allocation_p7z
    tree_sequestration_fixed_cost_p7z = annual_fixed_cost * cash_allocation_p7z
    tree_sequestration_fixed_wc_c0p7z = annual_fixed_cost * wc_allocation_c0p7z
    tree_sequestration_cashflow_p7zl = annual_cashflow_l * cash_allocation_p7z[...,na]
    tree_sequestration_wc_c0p7zl = annual_cashflow_l * wc_allocation_c0p7z[...,na]
    
    ##store r_vals
    fun.f1_make_r_val(r_vals, tree_sequestration_income_p7zl, 'tree_sequestration_income_p7zl', mask_season_p7z[...,na], z_pos=-1)
    fun.f1_make_r_val(r_vals, tree_sequestration_variable_cost_p7z, 'tree_sequestration_variable_cost_p7z', mask_season_p7z, z_pos=-1)
    fun.f1_make_r_val(r_vals, tree_sequestration_fixed_cost_p7z, 'tree_sequestration_fixed_cost_p7z', mask_season_p7z, z_pos=-1)

    fun.f1_make_r_val(r_vals, annual_sequestration_l, 'tree_co2_sequestration_l')
    fun.f1_make_r_val(r_vals, co2e_sold_l, 'tree_co2e_sold_l')
    fun.f1_make_r_val(r_vals, np.mean(co2e_fuel), 'tree_co2e_fuel')

    return tree_sequestration_cashflow_p7zl, tree_sequestration_wc_c0p7zl, tree_sequestration_fixed_cost_p7z, tree_sequestration_fixed_wc_c0p7z
    
    
def f_biodiversity(r_vals, mask_season_p7z):
    '''
    Calculate the per-hectare value of biodiversity credits from tree plantations.

    Assumptions:
    1. Biodiversity credit payments are received upfront at project commencement.
    2. The project is perpetual in nature (i.e., the biodiversity covenant remains indefinitely),
       so the annualised value is calculated using the perpetuity formula (A = NPV × r).
    '''
    project_duration = uinp.tree["controls"]["project_duration"]  # Total number of years
    biodiversity_included = uinp.tree["controls"]["include_biodiversity_credit"]
    biodiversity_costs = uinp.tree["biodiversity_costs"]
    plantation_structure = uinp.tree["controls"]["plantation_structure"]
    plantation_config = uinp.tree[f"plantation_structure_{plantation_structure}"]

    variable_costs_y = np.zeros(project_duration)
    fixed_costs_y = np.zeros(project_duration)
    total_credit_value_y = np.zeros(project_duration)
    
    #credits and costs per ha
    if biodiversity_included and plantation_config["biodiversity_value"]>0:
        total_credit_value_y[0] = plantation_config["biodiversity_value"]  #all credits recieved at the start of the project
        variable_costs_y[1:] = biodiversity_costs["annual_monitoring"]
        fixed_costs_y[0] = biodiversity_costs["setup"]

    discount_rate = uinp.finance['i_interest']
    npv_income = npf.npv(discount_rate, total_credit_value_y)
    npv_variable_cost = npf.npv(discount_rate, variable_costs_y)
    npv_fixed_cost = npf.npv(discount_rate, fixed_costs_y)

    # ---- Annualisation ----
    # Income: perpetual (land covenant)
    annual_income = npv_income * discount_rate

    # Costs: finite (limited to project duration)
    crf = discount_rate / (1 - (1 + discount_rate) ** -project_duration)
    annual_variable_cost = npv_variable_cost * crf
    annual_fixed_cost = npv_fixed_cost * crf

    annual_cashflow = annual_income - annual_variable_cost
    
    #allocate to p7
    cash_allocation_p7z, wc_allocation_c0p7z = fin.f_cashflow_allocation(np.array([uinp.tree["cost_date"]]), z_pos=-1)
    tree_biodiversity_income_p7z = annual_income * cash_allocation_p7z
    tree_biodiversity_variable_cost_p7z = annual_variable_cost * cash_allocation_p7z
    tree_biodiversity_fixed_cost_p7z = annual_fixed_cost * cash_allocation_p7z
    tree_biodiversity_fixed_wc_c0p7z = annual_fixed_cost * wc_allocation_c0p7z
    tree_biodiversity_cashflow_p7z = annual_cashflow * cash_allocation_p7z
    tree_biodiversity_wc_c0p7z = annual_cashflow * wc_allocation_c0p7z

    
    ##store r_vals
    fun.f1_make_r_val(r_vals, tree_biodiversity_income_p7z, 'tree_biodiversity_income_p7z', mask_season_p7z, z_pos=-1)
    fun.f1_make_r_val(r_vals, tree_biodiversity_variable_cost_p7z, 'tree_biodiversity_variable_cost_p7z', mask_season_p7z, z_pos=-1)
    fun.f1_make_r_val(r_vals, tree_biodiversity_fixed_cost_p7z, 'tree_biodiversity_fixed_cost_p7z', mask_season_p7z, z_pos=-1)

    return tree_biodiversity_cashflow_p7z, tree_biodiversity_wc_c0p7z, tree_biodiversity_fixed_cost_p7z, tree_biodiversity_fixed_wc_c0p7z
    
def f_deepflow(r_vals):
    recharge_l = pinp.tree["recharge_l"]
    fun.f1_make_r_val(r_vals, recharge_l, 'recharge_l')
    
def f1_tree_cashflow(r_vals):
    '''
    This function calculates the cashflow for the tree plantation enterprise.
    '''
    ###make z8 mask - used to uncluster
    date_season_node_p7z = per.f_season_periods()[:-1,...] #slice off end date p7
    mask_season_p7z = zfun.f_season_transfer_mask(date_season_node_p7z,z_pos=-1,mask=True)
    
    # Establishment & maint costs
    tree_estab_cost_p7zl, tree_estab_wc_c0p7zl = f_costs(r_vals, mask_season_p7z)
    
    # Income from harvest
    tree_biomass_cashflow_p7zl, tree_biomass_wc_c0p7zl = f_harvestable_biomass(r_vals, mask_season_p7z)
    
    # Income from carbon sequestration
    tree_sequestration_cashflow_p7zl, tree_sequestration_wc_c0p7zl, tree_sequestration_fixed_cost_p7z, tree_sequestration_fixed_wc_c0p7z = f_sequestration(r_vals, mask_season_p7z)
    
    # Income from biodiversity
    tree_biodiversity_cashflow_p7z, tree_biodiversity_wc_c0p7z, tree_biodiversity_fixed_cost_p7z, tree_biodiversity_fixed_wc_c0p7z = f_biodiversity(r_vals, mask_season_p7z)
    
    # Calculate total cashflow / wc
    tree_cashflow_p7zl = -tree_estab_cost_p7zl + tree_biomass_cashflow_p7zl + tree_sequestration_cashflow_p7zl + tree_biodiversity_cashflow_p7z[...,na]
    tree_wc_c0p7zl = -tree_estab_wc_c0p7zl + tree_biomass_wc_c0p7zl + tree_sequestration_wc_c0p7zl + tree_biodiversity_wc_c0p7z[...,na]

    # calculate total fixed costs/wc
    tree_fixed_cashflow_p7z = -(tree_sequestration_fixed_cost_p7z + tree_biodiversity_fixed_cost_p7z)
    tree_fixed_wc_c0p7z = -(tree_sequestration_fixed_wc_c0p7z + tree_biodiversity_fixed_wc_c0p7z)

    return tree_cashflow_p7zl, tree_wc_c0p7zl, tree_fixed_cashflow_p7z, tree_fixed_wc_c0p7z
    
    
      
##collates all the params
def f1_trees(params,r_vals):
    '''collates all the params'''
    tree_cashflow_p7zl, tree_wc_c0p7zl, tree_fixed_cashflow_p7z, tree_fixed_wc_c0p7z = f1_tree_cashflow(r_vals)
    f_deepflow(r_vals)
    
    keys_p7 = per.f_season_periods(keys=True)
    keys_c0 = sinp.general['i_enterprises_c0']
    keys_z = zfun.f_keys_z()
    keys_l = pinp.general['i_lmu_idx']

    arrays_p7zl = [keys_p7, keys_z, keys_l]
    arrays_p7z = [keys_p7, keys_z]
    arrays_c0p7zl = [keys_c0, keys_p7, keys_z, keys_l]
    arrays_c0p7z = [keys_c0, keys_p7, keys_z]

    params['p_tree_cashflow_p7zl'] = fun.f1_make_pyomo_dict(tree_cashflow_p7zl, arrays_p7zl)
    params['p_tree_wc_c0p7zl'] = fun.f1_make_pyomo_dict(tree_wc_c0p7zl, arrays_c0p7zl)
    params['p_tree_fixed_cashflow_p7z'] = fun.f1_make_pyomo_dict(tree_fixed_cashflow_p7z, arrays_p7z)
    params['p_tree_fixed_wc_c0p7z'] = fun.f1_make_pyomo_dict(tree_fixed_wc_c0p7z, arrays_c0p7z)

    