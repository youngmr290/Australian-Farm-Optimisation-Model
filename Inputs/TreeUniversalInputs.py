
import numpy as np

tree_inputs = {
    ###############
    # general
    ###############
    "controls": {
        "project_duration": 25,  # years
        "plantation_structure": 0,  # choose option from below
        "include_livestock_shelter": 0,  # 0 = no, 1 = yes
        "include_adjacent_pad_interaction": 0,  # 0 = no, 1 = yes
        "include_carbon_credit": 0,  # 0 = no, 1 = yes
        "include_biodiversity_credit": 0,  # 0 = no, 1 = yes
        "include_harvesting": 0,  # 0 = no, 1 = yes
    } , 
    ###############
    # Carbon block planting config
    ###############
    "plantation_structure_0": {
        "number_of_belts": 1,  #for block planting this should be 1
        "distance_between_belts": 500, # meters - for block plantins this should be the width of adjacent paddock.
        "propn_paddock_adj": 1,  # proportion of paddock adjacent to the belt. This should be less than 1 if planting next to bush or other non-arable area.
        "number_of_rows": 23,
        "between_row_spacing": 4.3,  # meters
        "within_row_spacing": 4.3,  # meters
        "side_buffer": 5,  # meters
        "biodiversity_credits": 2,  # credits/ha
        "shrub_density": 0,  # stems/ha

        # biomass green tonnes per hectare per year
        "biomass_harvested_y": np.array([
            0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
            0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
            0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
            0.00, 0.00
        ]),

        # Kg of CO2/ha sequestered in each yr from FullCAM. Slice 1 represents the end of yr1.
        "annual_sequestration": np.array([
            0, 905, 301, 2045, 4934, 7510, 9267, 10284, 10762, 10882,
            10771, 10517, 10176, 9788, 9376, 8958, 8543, 8138, 7747, 7373,
            7017, 6678, 6358, 6055, 5768, 5498
        ]),

        # crop / pasture production - Parameters represents the relative production by distance from the trees
        "protected_side_production_logistic_params": {
            #
            "offset": 0.0204,
            "k": 0.3474,
            "x0": 4.9304,
            "a": 8038.0492,
            "mu": -2583.9371,
            "sigma": 785.5683
        },
        "nonprotected_side_production_logistic_params": {
            "L": 1.0,
            "k": 0.4243,
            "x0": 4.8684,
            "offset": 0.1587
        },

        # Microclimate - Parameters represents the relative production by distance from the trees
        "ws_logistic_params": {
            "L": 1,
            "k": 0.0238,
            "x0": 34.4459,
            "offset": 0.0
        },
        "temp_logistic_params": {
            "L": 1,
            "k": 1000,
            "x0": 0,
            "offset": 1
        },

    },
    ###############
    # Carbon & co-benefits belt planting config
    ###############
    "plantation_structure_1": {
        "number_of_belts": 5,  #for block planting this should be 1
        "distance_between_belts": 150, # meters - for block plantins this should be the width of adjacent paddock.
        "propn_paddock_adj": 1,  # proportion of paddock adjacent to the belt. This should be less than 1 if planting next to bush or other non-arable area.
        "number_of_rows": 2,
        "between_row_spacing": 2.0,  # meters
        "within_row_spacing": 2.0,  # meters
        "side_buffer": 4,  # meters
        "biodiversity_credits": 1,  # credits/ha
        "shrub_density": 0,  # stems/ha

        # biomass green tonnes per hectare per year
        "biomass_harvested_y": np.array([
            0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
            0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
            0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
            0.00, 0.00
        ]),

        # Kg of CO2/ha sequestered in each yr from FullCAM. Slice 1 represents the end of yr1.
        "annual_sequestration": np.array([
            0, 1266, 422, 2863, 6908, 10514, 12974, 14397, 15067, 15235,
            15080, 14723, 14247, 13703, 13127, 12541, 11960, 11393, 10846, 10322,
            9823, 9349, 8901, 8476, 8076, 7698
        ]),

        # crop / pasture production - Parameters represents the relative production by distance from the trees
        "protected_side_production_logistic_params": {
            "offset": 0.0204,
            "k": 0.3474,
            "x0": 4.9304,
            "a": 8038.0492,
            "mu": -2583.9371,
            "sigma": 785.5683
        },
        "nonprotected_side_production_logistic_params": {
            "L": 1.0,
            "k": 0.4243,
            "x0": 4.8684,
            "offset": 0.1587
        },

        # Microclimate - Parameters represents the relative production by distance from the trees
        "ws_logistic_params": {
            "L": 1,
            "k": 0.0238,
            "x0": 34.4459,
            "offset": 0.0
        },
        "temp_logistic_params": {
            "L": 1,
            "k": 1000,
            "x0": 0,
            "offset": 1
        },

    },
    ###############
    # Biodiversity planting config
    ###############
    "plantation_structure_2": {
        "number_of_belts": 1,  #for block planting this should be 1
        "distance_between_belts": 500, # meters - for block plantins this should be the width of adjacent paddock.
        "propn_paddock_adj": 1,  # proportion of paddock adjacent to the belt. This should be less than 1 if planting next to bush or other non-arable area.
        "number_of_rows": 17,
        "between_row_spacing": 6.0,  # meters
        "within_row_spacing": 6.0,  # meters
        "side_buffer": 5,  # meters
        "biodiversity_credits": 5,  # credits/ha
        "shrub_density": 300,  # stems/ha

        # biomass green tonnes per hectare per year
        "biomass_harvested_y": np.array([
            0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
            0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
            0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
            0.00, 0.00
        ]),

        # Kg of CO2/ha sequestered in each yr from FullCAM. Slice 1 represents the end of yr1.
        "annual_sequestration": np.array([
            0, 901, 67, 435, 1463, 2780, 3970, 4867, 5465, 5817,
            5981, 6010, 5945, 5816, 5645, 5448, 5236, 5018, 4798, 4581,
            4370, 4166, 3970, 3782, 3604, 3435
        ]),

        # crop / pasture production - Parameters represents the relative production by distance from the trees
        "protected_side_production_logistic_params": {
            "offset": 0.0204,
            "k": 0.3474,
            "x0": 4.9304,
            "a": 8038.0492,
            "mu": -2583.9371,
            "sigma": 785.5683
        },
        "nonprotected_side_production_logistic_params": {
            "L": 1.0,
            "k": 0.4243,
            "x0": 4.8684,
            "offset": 0.1587
        },

        # Microclimate - Parameters represents the relative production by distance from the trees
        "ws_logistic_params": {
            "L": 1,
            "k": 0.0238,
            "x0": 34.4459,
            "offset": 0.0
        },
        "temp_logistic_params": {
            "L": 1,
            "k": 1000,
            "x0": 0,
            "offset": 1
        },

    },
    ###############
    # biomass planting config
    ###############
    "plantation_structure_3": {
        "number_of_belts": 5,  #for block planting this should be 1
        "distance_between_belts": 150, # meters - for block plantins this should be the width of adjacent paddock.
        "propn_paddock_adj": 1,  # proportion of paddock adjacent to the belt. This should be less than 1 if planting next to bush or other non-arable area.
        "number_of_rows": 2,
        "between_row_spacing": 2.0,  # meters
        "within_row_spacing": 2.0,  # meters
        "side_buffer": 4,  # meters
        "biodiversity_credits": 0,  # credits/ha
        "shrub_density": 0,  # stems/ha

        # biomass green tonnes per hectare per year
        "biomass_harvested_y": np.array([
            0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
            42.34, 0.00, 0.00, 0.00, 0.00, 0.00, 44.53, 0.00,
            0.00, 0.00, 0.00, 0.00, 46.23, 0.00, 0.00, 0.00,
            0.00, 42.59
        ]),

        # Kg of CO2/ha sequestered in each yr from FullCAM. Slice 1 represents the end of yr1.
        "annual_sequestration": np.array([
            0, 660, 4435, 13054, 17680, 18753, 18198, 17044, 15727, -46330,
            15481, 15449, 15053, 14534, 13934, -51932, 13459, 13974, 14034, 13884,
            13578, -54805, 13244, 13913, 14087, -33732
        ]),

        # crop / pasture production - Parameters represents the relative production by distance from the trees
        "protected_side_production_logistic_params": {
            "offset": 0.006,
            "k": 0.2956,
            "x0": 2.4377,
            "a": 0.0871,
            "mu": 12.1192,
            "sigma": 2.6905
        },
        "nonprotected_side_production_logistic_params": {
            "L": 1.0,
            "k": 0.3929,
            "x0": 3.8014,
            "offset": 0.1882
        },

        # Microclimate - Parameters represents the relative production by distance from the trees
        "ws_logistic_params": {
            "L": 1,
            "k": 0.1760,
            "x0": 5.9945,
            "offset": 0.0
        },
        "temp_logistic_params": {
            "L": 1,
            "k": 1000,
            "x0": 0,
            "offset": 1
        },

    },
    ###############
    # establishment & maint costs
    ###############
    "cost_date": 90, #March ish
    "initial_costs": {
        "site_prep": {
            "ripping_mounding": 280.00,  # $/ha
            "initial_weed_control": 54.00,  # $/ha
            "fertiliser_base": 0.00,  # $/ha
            "fertiliser_extra_if_harvested": 82.00,  # $/ha
        },
        "planting": {
            "mortality": 0.1,  # % that die and are replanted.
            "seedlings": 0.8,  # $/plant
            "planting_rate": 3000,  # plants/day that can be planted
            "planting_equipment": 300.00,  # $/day
            "farmer_labour_planting": 800.00,  # $/day
        },
    },
    "yr_1_costs": {
        "weed_control": 10.00,  # $/ha
        "fertiliser_base": 0.00,  # $/ha
        "fertiliser_extra_if_harvested": 82.00,  # $/ha
    },
    "yrs_2_to_100": {
        "weed_control": 0.00,  # $/ha
        "fertiliser_base": 0.00,  # $/ha
        "fertiliser_extra_if_harvested": 82.00,  # $/ha
    },

    ###############
    # Biodiversity
    ###############
    "biodiversity_costs": {
        "setup": 10000, #$/farm
        "annual_monitoring": 25, #$/ha
    },
    ###############
    # Carbon sequestration
    ###############
    "sequestration_costs": {
        "setup": 10000, #$/farm
        "annual_monitoring": 25, #$/ha
    },
    "fuel_used": {
        "initial": 40, # l/ha for site prep, spraying, fertilising and planting
        "yr1": 8,# l/ha for spraying & fertilising
        "yr2_to_100": 0, # l/ha for spraying & fertilising
    },
    "risk_of_reversal_buffer": 0.05, # a reserve pool of ACCUs withheld to cover risks like fire, drought, or non-compliance. Basically this is a propn of the estimiated ACCUs not converted to income for the farmer. 
    ###############
    # Biomass harvesting
    ###############
    "biomass_harvesting": {
        "contract_costs": 300, #$/hr
        "harvest_rate": 60, #green t/hr
        "transport_distance": 60, #kms to processor
        "transport_cost": 0.15, #$/gt/km
    },
    ###############
    # Prices
    ###############
    "carbon_price": 40,  # $/t co2e
    "biodiversity_credit_price": 2500,  # $/credit
    "biomass_price": 50,  # $/t green
}

#######################################################
#input calcs
#######################################################
if __name__ == '__main__':
    ########################################################
    #graph out the tree plantation structure
    ########################################################
    import matplotlib.pyplot as plt

    def plot_farm_layout(number_of_belts=5, between_row_spacing=3, within_row_spacing=2, side_buffer=1.5):
        number_of_rows=5  
    
        # Calculate belt width based on number of rows and spacing
        belt_width = (number_of_belts - 1) * between_row_spacing + 2 * side_buffer
        
        competition = 8
        open_padock = 20
        belt_start = open_padock + competition
        belt_end = belt_start + belt_width
        belt_center = (belt_start + belt_end) / 2 
        total_x = competition * 2 + open_padock * 2 + belt_width

        # Ensure the belt is at least 40% of the total x-axis by adjusting competition and open paddock zones proportionally
        # print(f"Initial values - Belt Start: {belt_start}, Belt End: {belt_end}, Belt Center: {belt_center}, Belt Width: {belt_width}, Total X: {total_x}, Competition: {competition}, Open Paddock: {open_padock}")
    
        belt_propn = 0.3
        min_belt_width = belt_propn * total_x
        
        if belt_width < min_belt_width:
            total_x_needed = belt_width / belt_propn  # Compute the required total_x to maintain proportion
            scale_factor = total_x_needed / total_x  # Compute scaling factor
            competition *= scale_factor
            open_padock *= scale_factor
            total_x = total_x_needed  # Update total_x to the new required value
            
            # print(f"Adjusted values - Belt Width: {belt_width}, Total X: {total_x}, Competition: {competition}, Open Paddock: {open_padock}")
            
            # Recalculate section positions
            belt_start = open_padock + competition
            belt_end = belt_start + belt_width
            belt_center = (belt_start + belt_end) / 2

        # print(f"Final values - Belt Start: {belt_start}, Belt End: {belt_end}, Belt Center: {belt_center}, Total X: {total_x}")

        # Define section boundaries dynamically
        sections = {
            "Open paddock": (0, open_padock),
            "Competition zone": (open_padock, belt_start),
            "Belt": (belt_start, belt_end),
            "Competition zone (Right)": (belt_end, belt_end + competition),
            "Open paddock (Right)": (belt_end + competition, belt_end + competition+open_padock)
        }
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(14, 8))

        # Add section labels
        for section, (start, end) in sections.items():
            ax.text((start + end) / 2, (number_of_rows+2) * within_row_spacing, section.replace(" (Right)", ""), ha='center', fontsize=14, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # Draw vertical dashed lines to separate sections
        for start, _ in sections.values():
            ax.axvline(x=start, color='black', linestyle="dashed")

        # Draw trees in the belt section based on row and spacing parameters
        tree_positions = []
        for i in range(number_of_belts):
            row_x = belt_start + side_buffer + i * between_row_spacing
            for j in range(number_of_rows):  # Creating 5 lines of trees
                tree_y = (number_of_rows) * within_row_spacing - (j * within_row_spacing)  # Adjusted to fit the taller figure
                tree_positions.append([row_x, tree_y])

        for x, y in tree_positions:
            ax.scatter(x, y, s=500, color="green", label="Tree")

        # Adjust label positions dynamically
        between_row_label_y = max(y for _, y in tree_positions) + within_row_spacing/2
        buffer_label_y = min(y for _, y in tree_positions) - within_row_spacing/2
        within_row_label_y = max(y for _, y in tree_positions) - within_row_spacing/2
        end_tree_pos = max(x for x, _ in tree_positions)
        
        # Add inter-row space label above trees
        ax.text(belt_center, between_row_label_y + 0.5, f"Between-row space ({between_row_spacing}m)",
            ha='center', fontsize=12, fontweight='bold')
        ax.annotate("", xy=(belt_center-between_row_spacing/2, between_row_label_y), xytext=(belt_center+between_row_spacing/2, between_row_label_y),
                arrowprops=dict(arrowstyle='<->', lw=1.5))

        # Add within-row space label between inter-row spaces
        ax.text(belt_center, within_row_label_y - 0.5, f"Within-row space ({within_row_spacing}m)",
            ha='center', fontsize=12, fontweight='bold')
        ax.annotate("", xy=(belt_start+side_buffer+between_row_spacing/2, within_row_label_y-within_row_spacing/2), xytext=(belt_start+side_buffer+between_row_spacing/2, within_row_label_y+within_row_spacing/2),
                arrowprops=dict(arrowstyle='<->', lw=1.5))

        # Add buffer space label at bottom
        ax.text(belt_center, buffer_label_y, f"Side-buffer ({side_buffer}m)",
            ha='center', fontsize=12, fontweight='bold')
        ax.annotate("", xy=(belt_end-side_buffer, buffer_label_y), xytext=(belt_end, buffer_label_y),
                arrowprops=dict(arrowstyle='<->', lw=1.5))

        # Add wind direction annotation at belt center
        ax.annotate("Wind direction", xy=(50, (number_of_rows+1.5) * within_row_spacing), xytext=(belt_center, (number_of_rows+1.5) * within_row_spacing),
                    ha='center', fontsize=12, arrowprops=dict(arrowstyle='->'))

        # Adjust axes
        ax.set_ylim(0, (number_of_rows+2) * within_row_spacing)  # Increased limit for better spacing
        ax.set_xlim(0, total_x)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

        # Show figure
        plt.show()

    # Example usage
    config = tree_inputs[f"plantation_structure_{tree_inputs['controls']['plantation_structure']}"]
    # plot_farm_layout(config["number_of_rows"], config["between_row_spacing"], config["within_row_spacing"], config["side_buffer"])


    
    ########################################################
    #fit a curve to production data and windspeed data
    ########################################################

    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from scipy.special import erf

    def fit_gauss_tail(x_data, y_data, p0=None, show_plot=True):
        """
        Function is a combination of a logistic function and a Gaussian bump, adjusted to ensure the tail returns to 1.

        Fit the function:
        Y(x) = Y_normal + a*exp(-((x-b)/c)**2) - p/(x+q)
        to the provided data (x_data, y_data).
        
        Parameters
        ----------
        x_data : array-like
            The x-values of your data.
        y_data : array-like
            The y-values of your data.
        p0 : list or None
            Optional initial guesses for [Y_normal, a, b, c, p, q].
            If None, uses a default set of guesses.
        show_plot : bool
            If True, display a plot of the data and the fitted curve.
        
        Returns
        -------
        params : array
            The best-fit parameters in the order [Y_normal, a, b, c, p, q].
        cov : 2D array
            The covariance matrix from curve_fit.
        """
        
        # Define the model
        def gaussian_bump_return_to_one(x, offset, k, x0, a, mu, sigma):
            logistic_part = offset + (1 - offset) / (1 + np.exp(-k * (x - x0)))
            bump = a * np.exp(-((x - mu) / sigma) ** 2)
            return logistic_part + bump - bump[-1]  # ensures tail returns to 1

        # If no initial guesses are provided, pick some defaults
        if p0 is None:
            p0 = [0.2, 0.5, 5.0, 0.1, 15.0, 10.0]  # offset, k, x0, a, mu, sigma

        # Fit the model to the data
        try:
            params, cov = curve_fit(
                gaussian_bump_return_to_one,
                x_data,
                y_data,
                p0=p0,
                maxfev=20000
            )
        except RuntimeError as e:
            print("Fit failed:", e)
            return None, None

        if show_plot:
            x_fit = np.linspace(min(x_data), max(x_data), 300)
            y_fit = gaussian_bump_return_to_one(x_fit, *params)
            plt.figure(figsize=(6, 4))
            plt.scatter(x_data, y_data, color='red', label='Data')
            plt.plot(x_fit, y_fit, color='green', label='Custom Fit')
            plt.xlabel('Distance from Trees (m)')
            plt.ylabel('Relative Windspeed')
            plt.title('Custom Fit: Logistic + Gaussian Bump')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        # Print the results
        offset_fit, k_fit, x0_fit, a_fit, mu_fit, sigma_fit = params
        print("Fitted parameters:")
        print(f"  offset = {offset_fit:.4f}")
        print(f"  k      = {k_fit:.4f}")
        print(f"  x0     = {x0_fit:.4f}")
        print(f"  a      = {a_fit:.4f}")
        print(f"  mu     = {mu_fit:.4f}")
        print(f"  sigma  = {sigma_fit:.4f}")
        
        return params, cov


    def fit_logistic_model(x_data, y_data, p0=None, show_plot=True):
        """
        Fit a logistic function of the form:

            Y(x) = offset + (1 - offset) / [1 + exp(-k*(x - x0))]

        This version fixes the upper asymptote (L) to 1.

        Returns:
            Best-fit parameters [k, x0, offset]
        """

        # Define logistic with fixed L = 1
        def logistic_fixed_L(x, k, x0, offset):
            return offset + (1 - offset) / (1 + np.exp(-k * (x - x0)))

        # Initial guess
        if p0 is None:
            offset_guess = min(y_data)
            x0_guess = 0#np.median(x_data)
            k_guess = 0.1
            p0 = [k_guess, x0_guess, offset_guess]

        # Bounds to help convergence
        bounds = (
            [0.001, 0, 0],  # lower bounds: k, x0, offset
            [10.0, 500, 0.99]  # upper bounds: k, x0, offset
        )

        try:
            params, cov = curve_fit(
                logistic_fixed_L,
                x_data,
                y_data,
                p0=p0,
                bounds=bounds,
                maxfev=20000
            )
        except RuntimeError as e:
            print("Fit failed:", e)
            return None, None

        k_fit, x0_fit, offset_fit = params

        # Plot
        if show_plot:
            x_fit = np.linspace(min(x_data), max(x_data), 300)
            y_fit = logistic_fixed_L(x_fit, k_fit, x0_fit, offset_fit)
            plt.figure(figsize=(6, 4))
            plt.scatter(x_data, y_data, color='red', label='Data')
            plt.plot(x_fit, y_fit, color='blue', label='Fit (L=1)')
            plt.title('Logistic Fit with L=1')
            plt.legend()
            plt.tight_layout()
            plt.show()

        print("Fitted logistic parameters (L = 1):")
        print(f"  k      = {k_fit:.4f}")
        print(f"  x0     = {x0_fit:.4f}")
        print(f"  offset = {offset_fit:.4f}")

        return params, cov
    # --------------------------------
    # Get parameters for WS model
    # --------------------------------
    x_data = np.array([0, 10, 30, 50, 100, 150, 200, 250, 300])

    # Windspeed data by configuration
    config_data = {
        "Config 1": np.array([0.25, 0.35, 0.5, 0.65, 0.80, 0.90, 0.94, 0.98, 1.00]),
        "Config 2": np.array([0.25, 0.35, 0.5, 0.65, 0.80, 0.90, 0.94, 0.98, 1.00]),
        "Config 3": np.array([0.25, 0.35, 0.5, 0.65, 0.80, 0.90, 0.94, 0.98, 1.00]),
        "Config 4": np.array([0.25, 0.68, 0.93, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00])
    }

    # Fit and store results
    fit_results = {}

    for name, y_data in config_data.items():
        print(f"\nFitting logistic model for {name}")
        params, cov = fit_logistic_model(x_data, y_data, show_plot=True)
        fit_results[name] = {
            "params": params,
            "covariance": cov
        }

    # ---------------------------
    # Get parameters for yield impact non-protected side
    # ---------------------------
    x_data = np.array([0, 2.75, 5.5, 8.25, 11, 20, 40])
    config_data = {
        "Config 1": np.array([0.24, 0.43, 0.62, 0.81, 1.00, 1.00, 1.00]),
        "Config 2": np.array([0.24, 0.43, 0.62, 0.81, 1.00, 1.00, 1.00]),
        "Config 3": np.array([0.24, 0.43, 0.62, 0.81, 1.00, 1.00, 1.00]),
        "Config 4": np.array([0.33, 0.53, 0.71, 0.86, 1.00, 1.00, 1.00])
    }

    # Fit and store results
    fit_results = {}

    for name, y_data in config_data.items():
        print(f"\nFitting logistic model for {name}")
        params, cov = fit_logistic_model(x_data, y_data, show_plot=True, p0 = [1.0, 5.5, 0.24])
        fit_results[name] = {
            "params": params,
            "covariance": cov
        }

    # ---------------------------
    # Get parameters for yield impact protected side
    # ---------------------------
    x_data = np.array([0, 2.75, 5.5, 8.25, 11, 15, 20, 30, 40, 80])
    config_data = {
        "Config 1": np.array([0.24, 0.43, 0.62, 0.81, 1.00, 1.02, 1.04, 1.04, 1.04, 1.00]),
        "Config 2": np.array([0.24, 0.43, 0.62, 0.81, 1.00, 1.02, 1.04, 1.04, 1.04, 1.00]),
        "Config 3": np.array([0.24, 0.43, 0.62, 0.81, 1.00, 1.02, 1.04, 1.04, 1.04, 1.00]),
        "Config 4": np.array([0.33, 0.53, 0.71, 0.86, 1.00, 1.004, 1.008, 1.008, 1.008, 1.00])
    }

    # Fit and store results
    fit_results = {}

    for name, y_data in config_data.items():
        print(f"\nFitting logistic model for {name}")
        params, cov = fit_gauss_tail(x_data, y_data, show_plot=True)
        fit_results[name] = {
            "params": params,
            "covariance": cov
        }
