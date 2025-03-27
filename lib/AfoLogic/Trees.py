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
from . import Finance as fin
from . import Functions as fun


na=np.newaxis


def f_costs():
    '''
    The main production costs are:

        - Establishment and maintenance costs for the trees
        - Costs for harvest and on-farm transport
        - Road transport to the bioenergy plant
        - Costs for auditing and administration.

    Note, the opportunity cost of the land that is affected by the trees is considered in the other functions.
    Note 2, labour is a cost rather than being hooked up to the labour constraints because it is only a once off thing (and may be completed by contractors).
    '''

    # Retrieve selected plantation structure
    config = uinp.tree[f"plantation_structure_{uinp.tree['controls']['plantation_structure']}"]
    include_harvesting = uinp.tree["controls"]["include_harvesting"]
    
    # Calculate the effective width (total area used by tree rows, including side buffers)
    effective_width = (config["number_of_rows"] - 1) * config["between_row_spacing"] + 2 * config["side_buffer"]

    # Calculate planting density (plants per hectare)
    planting_density = 10000 / effective_width / config["within_row_spacing"] * config["number_of_rows"]  # plants per hectare
    
    # Get LMU data as NumPy arrays
    tree_area_l = pinp.tree["area_trees_l"]
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
    
    # Compute total costs using NumPy sum (faster than Python sum)**
    total_establishment_costs = np.sum(establishment_costs_l * tree_area_l)
    total_yr1_costs = np.sum(yr1_costs_l * tree_area_l)
    total_yrs_2_to_100_costs = np.sum(yrs_2_to_100_costs_l * tree_area_l)
    
    # Discounted present value of future maintenance costs 
    discount_rate  = uinp.finance['i_interest']
    num_years = 100  # Total number of years

    # Construct cash flow array
    total_cash_flows = [-total_establishment_costs, -total_yr1_costs] + [-total_yrs_2_to_100_costs] * (num_years - 1)
    
    # Compute NPV 
    npv = npf.npv(discount_rate, total_cash_flows)
    
    # Compute annualized cost as an annuity
    annual_payment = npv * (discount_rate / (1 - (1 + discount_rate) ** -num_years))
    
    #allocate to p7
    cost_allocation_p7z, wc_allocation_c0p7z = fin.f_cashflow_allocation(np.array([uinp.tree["cost_date"]]), z_pos=-1)
    tree_estab_cost_p7z = annual_payment * cost_allocation_p7z
    tree_estab_wc_c0p7z = annual_payment * wc_allocation_c0p7z
    return tree_estab_cost_p7z, tree_estab_wc_c0p7z
    
    
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

    # Get area of trees (in hectares) as a numpy array
    tree_area_l = pinp.tree["area_trees_l"]

    # Calculate the effective width of the tree rows (including side buffers)
    effective_width = ((plantation_config["number_of_rows"] - 1) * plantation_config["between_row_spacing"] +
                    2 * plantation_config["side_buffer"])

    # Calculate the length of each belt (in meters)
    belt_length_l = (tree_area_l * 10000) / (effective_width * number_of_belts) # Convert tree area from hectares to m² (1 ha = 10,000 m²)

    # Calculate LMU area available for production (excluding area with trees)
    lmu_area_without_trees_l = pinp.general['i_lmu_area'] - tree_area_l

    # Calculate the land area on the protected side of the tree belts
    protected_side_area_l = (belt_length_l * number_of_belts * distance_between_belts * propn_trees_adjacent) / 10000 # Convert m² to hectares

    # Calculate the fraction of the available LMU area (excluding tree area) that is on the protected side
    propn_lmu_protected_l = np.minimum(1, protected_side_area_l  / lmu_area_without_trees_l) # Ensure the fraction is not greater than 1 (cant shelter more than 100% of lmu)   

    # Assuming symmetry, the fraction on the non-protected side of trees is the same
    propn_lmu_nonprotected_l = propn_lmu_protected_l

    #########################################
    # Production on the PROTECTED side
    #########################################

    # Logistic parameters for the protected side (benefiting from wind protection and incurring resource competition)
    protected_side_params = uinp.tree["protected_side_production_logistic_params"]
    L_fit_protected = protected_side_params["L"]
    k_fit_protected = protected_side_params["k"]
    x0_fit_protected = protected_side_params["x0"]
    offset_fit_protected = protected_side_params["offset"]

    # Evaluate the indefinite logistic integral at 0 and at the distance between belts
    F0_protected = fun.f_logistic_integral(0, L_fit_protected, k_fit_protected, x0_fit_protected, offset_fit_protected)
    Fx_protected = fun.f_logistic_integral(distance_between_belts, L_fit_protected, k_fit_protected, x0_fit_protected, offset_fit_protected)

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
    nonprotected_side_params = uinp.tree["nonprotected_side_production_logistic_params"]
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
    
    return average_production_scalar_l



def f_microclimate_adj():
    '''
    This function accounts for microclimate effects of trees. The function
    calculates a windspeed scalar in paddocks adjacent to trees. The scalar is used to adjust the calculation of livestock chill.

    This assumes that livestock do not seek shelter. 
    Assumes that benefits of shelter only exist in the paddock adjacent to trees

    Chill is a non-linear function (and WS by distance from trees funcion). 
    Therefore, an improvement would be to report the microclimate adjusters for different parts of the paddock.

    '''
    # Get plantation configuration based on the plantation structure control setting
    plantation_structure = uinp.tree["controls"]["plantation_structure"]
    plantation_config = uinp.tree[f"plantation_structure_{plantation_structure}"]

    # Extract configuration values
    distance_between_belts = plantation_config["distance_between_belts"]  # meters between tree belts or to the edge of the paddock
    number_of_belts = plantation_config["number_of_belts"]
    propn_trees_adjacent = plantation_config["propn_paddock_adj"]  # Proportion of trees adjacent to usable paddock

    # Get total area of trees - don't need to differentiate between LMUs
    tree_area = np.sum(pinp.tree["area_trees_l"]) 

    # Calculate the effective width of the tree rows (including side buffers)
    effective_width = ((plantation_config["number_of_rows"] - 1) * plantation_config["between_row_spacing"] +
                    2 * plantation_config["side_buffer"])

    # Calculate the length of each belt (in meters)
    belt_length = (tree_area * 10000) / (effective_width * number_of_belts) # Convert tree area from hectares to m² (1 ha = 10,000 m²)

    # Calculate the land area on the protected side of the tree belts
    protected_area = (belt_length * number_of_belts * distance_between_belts * propn_trees_adjacent) / 10000 # Convert m² to hectares

    #########################################
    # WS adjuster on the PROTECTED side
    #########################################

    # Logistic parameters for the protected side (benefiting from wind protection and incurring resource competition)
    ws_logistic_params = uinp.tree["ws_logistic_params"]
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

    # Logistic parameters for the protected side (benefiting from wind protection and incurring resource competition)
    temp_logistic_params = uinp.tree["temp_logistic_params"]
    L_fit_temp = temp_logistic_params["L"]
    k_fit_temp = temp_logistic_params["k"]
    x0_fit_temp = temp_logistic_params["x0"]
    offset_fit_temp = temp_logistic_params["offset"]

    # Evaluate the indefinite logistic integral at 0 and at the distance between belts
    F0_temp = fun.f_logistic_integral(0, L_fit_temp, k_fit_temp, x0_fit_temp, offset_fit_temp)
    Fx_temp = fun.f_logistic_integral(distance_between_belts, L_fit_temp, k_fit_temp, x0_fit_temp, offset_fit_temp)

    # Compute the relative production adjustment over the belt width
    relative_temp_adj = (Fx_temp - F0_temp) / distance_between_belts

    return relative_ws_adj, relative_temp_adj, protected_area



def f_harvestable_biomass():
    '''
    Calculate the income generated from harvesting tree biomass for biofuel.
    '''
    project_duration = uinp.tree["controls"]["project_duration"]  # Total number of years
    harvested = uinp.tree["controls"]["include_harvesting"]
    biomass_harvesting = uinp.tree["biomass_harvesting"]
    biomass_price = uinp.tree["biomass_price"]

    #biomass income and costs per ha
    if harvested:
        # income
        biomass_harvested_y = uinp.tree["biomass_harvested_y"][0:project_duration]
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
        biomass_income_yl = np.zeros([project_duration,1])
        costs_yl = np.zeros([project_duration,1])
        
    # Construct cash flow array
    tree_area_l = pinp.tree["area_trees_l"] 
    total_cash_flows_yl = (biomass_income_yl - costs_yl) * tree_area_l
    total_cash_flows_y = np.sum(total_cash_flows_yl, axis=-1)

    # Compute NPV 
    discount_rate  = uinp.finance['i_interest']
    npv = npf.npv(discount_rate, total_cash_flows_y)
    
    # Compute annualized cost as an annuity
    annual_payment = npv * (discount_rate / (1 - (1 + discount_rate) ** -project_duration))
    
    #allocate to p7
    cash_allocation_p7z, wc_allocation_c0p7z = fin.f_cashflow_allocation(np.array([uinp.tree["cost_date"]]), z_pos=-1)
    tree_biomass_cashflow_p7z = annual_payment * cash_allocation_p7z
    tree_biomass_cashflow_wc_c0p7z = annual_payment * wc_allocation_c0p7z
    return tree_biomass_cashflow_p7z, tree_biomass_cashflow_wc_c0p7z
  
  
  


def f_sequestration():
    '''
    Carbon sequestration from trees
    
    Note: This is assuming that the financial values in MIDAS are in real terms
    & will remain constant. In other words the opportunity cost of forgone
    agricultural production will be the same every yr in real terms.
    Hence the carbon price is also specfied in real terms.
    
    #TODO: check the info below if we use this method.
    To translate future income from sequestration into a form compatible with the steady state conditions of MIDAS, predictions of future sequestration (minus deductions for emissions) are multiplied by the projected C price in the yr of sequestration to give an estimate of sequestration payments in future years.
    These payments are then summed & discounted (ie turned into a NPV), & then annualised over the time sequestering land use occurs for (assumed to be 100 yrs due to permanency req'ments), giving the equivalent annual income for C sequestration that would be expected on a steady-state basis.
    However, MIDAS is setup to use a biophysical metric (kg CO2-e) to account for sequestration benefits rather than a financial metric ($ for sequestration per ha). Therefore, on the RH side of this Table, the financial annualised payments for sequestration are converted into an equivalent biophysical annual rate
    of sequestration, such that using this resultant biophysical metric has the same effect economically (i.e., when the model is run) as if the financial metric for sequestration benefits had been used instead. This biophysical annual rate of sequestration is what ultimately goes into the MPS.


    '''

    #plantation duration and risk buffer
    #need to conside fuel emissions

    #maybe need to consider time value of money like in MIDAS (ask dad about this)

    #estimate with FULLCAM
    
   #comments from midas - need to update 
    # This is the default estimate of sequestration in study region as predicted by the Aust. Gov's National Carbon Accounting Toolbox (NCAT) FullCAM model (ver 3.55) for unharvested mallee plantings grown in block configuration with rows 4 m apart, and 2m between each tree along a row. 
    # As trees are planted in block plantings (as opposed to belts with areas agriculture production), the detrimental effect of tree competition on agricultural production is ignored.
    # Sequestration has been estimated according to CFI methodologies where the carbon contained in living below- & above-ground biomass plus litter & debris is counted toward sequestration.
    # CFI permanency rules (as of 2013 at least) req sequestered carbon to be stored for 100 years after credits are first claimed. Subsequent claims made for the same project do not ‘reset’ the 100-year count, meaning that carbon sequestered later needs to only be stored for progressively less time. There is no penalty if sequestration is released after this 100 yr period. Therefore sequestration has been estimated assuming the plantings will be maintained for 100 yrs and all sequestration during this period will be claimed.

    
    
def f_biodiversity():
    '''
    Calculate the value of biodiversity credits from tree plantation.
    
    Assumption is that no biodiversity credits are avaliable if biomass is harvested.
    '''
    #TODO need to work out how payment structure works. If it is a propn per yr then need to discount
    #TODO how does plantation duration impact biodiversity credits???
    project_duration = uinp.tree["controls"]["project_duration"]  # Total number of years
    harvested = uinp.tree["controls"]["include_harvesting"]
    biodiversity_included = uinp.tree["controls"]["include_biodiversity_credit"]
    biodiversity_costs = uinp.tree["biodiversity_costs"]
    plantation_structure = uinp.tree["controls"]["plantation_structure"]
    plantation_config = uinp.tree[f"plantation_structure_{plantation_structure}"]

    costs_y = np.zeros(project_duration)
    biodiversity_credits_y = np.zeros(project_duration)
    
    #credits and costs per ha
    if biodiversity_included and not harvested:
        biodiversity_credits_y[0] = plantation_config["biodiversity_credits"]  #all credits recieved at the start of the project
        costs_y[0] = biodiversity_costs["setup"]
        costs_y[1:] = biodiversity_costs["annual_monitoring"] 

    # Compute the total value (dollars)
    tree_area = np.sum(pinp.tree["area_trees_l"]) 
    market_price_per_credit = uinp.tree["biodiversity_credit_price"]
    total_credit_value_y = biodiversity_credits_y * tree_area * market_price_per_credit
    
    # total costs accosiated with accessing biodiversity credits
    total_costs_y = costs_y * tree_area
    
    # Compute NPV 
    discount_rate  = uinp.finance['i_interest']
    total_cash_flows_y = total_credit_value_y - total_costs_y
    npv = npf.npv(discount_rate, total_cash_flows_y)
    
    # Compute annualized cost as an annuity
    annual_payment = npv * (discount_rate / (1 - (1 + discount_rate) ** -project_duration))
    
    #allocate to p7
    cash_allocation_p7z, wc_allocation_c0p7z = fin.f_cashflow_allocation(np.array([uinp.tree["cost_date"]]), z_pos=-1)
    tree_biodiversity_cashflow_p7z = annual_payment * cash_allocation_p7z
    tree_biodiversity_cashflow_wc_c0p7z = annual_payment * wc_allocation_c0p7z
    
    return tree_biodiversity_cashflow_p7z, tree_biodiversity_cashflow_wc_c0p7z
    
    
    
def f_tree_cashflow(r_vals, params):
    '''
    This function calculates the cashflow for the tree plantation enterprise.
    '''
    # Establishment & maint costs
    tree_estab_cost_p7z, tree_estab_wc_c0p7z = f_costs()
    
    # Income from harvest
    f_harvestable_biomass()
    
    # Income from carbon sequestration
    f_sequestration()
    
    # Income from biodiversity
    f_biodiversity()
    
    # Calculate total cashflow
    tree_cashflow_p7z = tree_estab_cost_p7z
    return tree_cashflow_p7z
    
    
      
##collates all the params
def f_trees(params,r_vals):
    f_microclimate_adj()
    f_tree_cashflow(r_vals, params)
    #deepflow - maybe this is a table by land use by LMU. Then just tally in pyomo.