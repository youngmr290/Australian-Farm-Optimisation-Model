
import numpy as np

tree_inputs = {
    ###############
    # general
    ###############
    "controls": {
        "project_duration": 100,  # years
        "plantation_structure": 0,  # 0 = dual row, 1 = wide spread belts
        "include_carbon_credit": 0,  # 0 = no, 1 = yes
        "include_biodiversity_credit": 0,  # 0 = no, 1 = yes
        "include_harvesting": 0,  # 0 = no, 1 = yes
    } , 
    ###############
    # plantation config
    ###############
    # Belt planting
    "plantation_structure_0": {
        "number_of_belts": 5,  #for block planting this should be 1
        "distance_between_belts": 100, # meters - for block plantins this should be the width of adjacent paddock.
        "propn_paddock_adj": 1,  # proportion of paddock adjacent to the belt. This should be less than 1 if planting next to bush or other non-arable area.
        "number_of_rows": 2,
        "between_row_spacing": 2.0,  # meters
        "within_row_spacing": 2.0,  # meters
        "side_buffer": 1.5,  # meters
        "biodiversity_credits": 5,  # credits/ha
    },
    #block planting
    "plantation_structure_1": {  # Wide-spread belts structure
        "number_of_belts": 1,  #for block planting this should be 1
        "distance_between_belts": 100, # meters - for block plantins this should be the width of adjacent paddock.
        "propn_paddock_adj": 0.5,  # proportion of paddock adjacent to the belt. This should be less than 1 if planting next to bush or other non-arable area.
        "number_of_rows": 5,
        "between_row_spacing": 6.0,  # meters
        "within_row_spacing": 3.0,  # meters
        "side_buffer": 3.0,  # meters
        "biodiversity_credits": 5,  # credits/ha
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
    # crop / pasture production
    ###############
    "protected_side_production_logistic_params": {  # Parameters for the logistic model for the sheltered side of the tree belt that receives the benefits of wind protection and the costs of resource competition. (this represents the relative production by distance from the trees)
        "L": 0.9999,
        "k": 0.2310,
        "x0": 12.4276,
        "offset": 0.6787
    },
    "nonprotected_side_production_logistic_params": {  # Parameters for the logistic model for the non-sheltered side of the tree belt that only receives the costs of resource competition. (this represents the relative production by distance from the trees)
        "L": 0.9999,
        "k": 0.2310,
        "x0": 12.4276,
        "offset": 0.6787
    },
    ###############
    # Micro climate
    ###############
    "ws_logistic_params": {  # Parameters for the logistic model (this represents the relative production by distance from the trees)
        "L": 0.9999,
        "k": 0.2310,
        "x0": 12.4276,
        "offset": 0.6787
    },
    "temp_logistic_params": {  # Parameters for the logistic model (this represents the relative production by distance from the trees)
        "L": 0.9999,
        "k": 0.2310,
        "x0": 12.4276,
        "offset": 0.6787
    },
    ###############
    # Biodiversity
    ###############
    "biodiversity_costs": {
        "setup": 40, #$/ha
        "annual_monitoring": 25, #$/ha
    },
    ###############
    # Carbon sequestration
    ###############
    "sequestration_costs": {
        "setup": 40, #$/ha
        "annual_monitoring": 25, #$/ha
    },
    ###############
    # Biomass harvesting
    ###############
    "biomass_harvesting": {
        "contract_costs": 300, #$/hr
        "harvest_rate": 60, #t/hr
        "transport_distance": 60, #kms to processor
        "transport_cost": 0.15, #$/t/km
    },
    "biomass_harvested_y": np.array([
        0, 0, 0, 0, 5,
        0, 0, 0, 0, 5,
        0, 0, 0, 0, 5,
        0, 0, 0, 0, 5,
        0, 0, 0, 0, 5,
        0, 0, 0, 0, 5,
        0, 0, 0, 0, 5,
        0, 0, 0, 0, 5,
        0, 0, 0, 0, 5,
        0, 0, 0, 0, 5,
        0, 0, 0, 0, 5,
        0, 0, 0, 0, 5,
        0, 0, 0, 0, 5,
        0, 0, 0, 0, 5,
        0, 0, 0, 0, 5,
        0, 0, 0, 0, 5,
        0, 0, 0, 0, 5,
        0, 0, 0, 0, 5,
        0, 0, 0, 0, 5,
        0, 0, 0, 0, 5
        ]),
    ###############
    # Prices
    ###############
    "biodiversity_credit_price": 200,  # $/credit
    "biomass_price": 50,  # $/t
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
    plot_farm_layout(config["number_of_rows"], config["between_row_spacing"], config["within_row_spacing"], config["side_buffer"])


    
    ########################################################
    #fit a curve to production data and windspeed data
    ########################################################

    
    #todo this these functions may change depending on the data. For now I have just used some random data and fitted an equation to it (this could happen outside of AFO).
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from scipy.special import erf

    def fit_gauss_tail(x_data, y_data, p0=None, show_plot=True):
        """
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
        def model(x, Y_normal, a, b, c, p, q):
            return Y_normal + a * np.exp(-((x - b)/c)**2) - p/(x + q)
        
        # If no initial guesses are provided, pick some defaults
        if p0 is None:
            # You can customize these defaults as needed
            p0 = [1.0,   # Y_normal ~ 1.0
                0.4,   # a        ~ 0.4
                20.0,  # b        ~ 20
                15.0,  # c        ~ 15
                0.3,   # p        ~ 0.3
                1.0]   # q        ~ 1.0
        
        # Fit the model to the data
        params, cov = curve_fit(model, x_data, y_data, p0=p0, maxfev=10000)
        Y_normal_fit, a_fit, b_fit, c_fit, p_fit, q_fit = params
        
        # Create a smooth range of x-values for plotting
        x_fit = np.linspace(np.min(x_data), np.max(x_data), 300)
        y_fit = model(x_fit, *params)
        
        # Plot if requested
        if show_plot:
            plt.figure(figsize=(6,4))
            plt.scatter(x_data, y_data, color='red', label='Data')
            plt.plot(x_fit, y_fit, color='blue', label='Fitted Curve')
            plt.xlabel('x')
            plt.ylabel('y(x)')
            plt.title(r'Fit: $Y(x)=Y_{\mathrm{normal}}+a\,e^{-((x-b)/c)^2}-\frac{p}{x+q}$')
            plt.legend()
            plt.tight_layout()
            plt.show()
        
        # Print the results
        print("Fitted parameters:")
        print(f"  Y_normal = {Y_normal_fit:.4f}")
        print(f"  a        = {a_fit:.4f}")
        print(f"  b        = {b_fit:.4f}")
        print(f"  c        = {c_fit:.4f}")
        print(f"  p        = {p_fit:.4f}")
        print(f"  q        = {q_fit:.4f}")
        
        return params, cov


    def fit_logistic_model(x_data, y_data, p0=None, show_plot=True):
        """
        Fit a logistic (sigmoid) function of the form:
        
            Y(x) = offset + (L - offset) / [1 + exp(-k*(x - x0))]
        
        to the provided data (x_data, y_data).
        
        Parameters
        ----------
        x_data : array-like
            The x-values of your data.
        y_data : array-like
            The y-values of your data.
        p0 : list or None
            Optional initial guesses for [L, k, x0, offset].
            If None, uses a default set of guesses.
        show_plot : bool
            If True, display a plot of the data and the fitted curve.
        
        Returns
        -------
        params : array
            Best-fit parameters [L, k, x0, offset].
        cov : 2D array
            Covariance matrix from curve_fit.
        """
        
        # Define logistic (sigmoid) model
        def logistic(x, L, k, x0, offset):
            return offset + (L - offset) / (1.0 + np.exp(-k * (x - x0)))
        
        # Provide default guesses if none are given
        if p0 is None:
            L_guess      = max(y_data)         # upper plateau near max of data
            offset_guess = min(y_data)         # lower plateau near min of data
            x0_guess     = np.median(x_data)   # midpoint around the median of x_data
            k_guess      = 0.1                 # slope guess
            p0 = [L_guess, k_guess, x0_guess, offset_guess]
        
        # Curve fit
        params, cov = curve_fit(logistic, x_data, y_data, p0=p0, maxfev=10000)
        L_fit, k_fit, x0_fit, offset_fit = params
        
        # Generate a smooth curve for plotting
        x_fit = np.linspace(np.min(x_data), np.max(x_data), 300)
        y_fit = logistic(x_fit, L_fit, k_fit, x0_fit, offset_fit)
        
        # Plot if requested
        if show_plot:
            plt.figure(figsize=(6,4))
            plt.scatter(x_data, y_data, color='red', label='Data')
            plt.plot(x_fit, y_fit, color='blue', label='Logistic Fit')
            plt.xlabel('x')
            plt.ylabel('y(x)')
            plt.title('Logistic (Sigmoid) Fit')
            plt.legend()
            plt.tight_layout()
            plt.show()
        
        # Print the results
        print("Fitted logistic parameters:")
        print(f"  L      = {L_fit:.4f}")
        print(f"  k      = {k_fit:.4f}")
        print(f"  x0     = {x0_fit:.4f}")
        print(f"  offset = {offset_fit:.4f}")
        
        return params, cov


    # --------------------------------
    # Get parameters for the logistic model
    # --------------------------------
    # Example data that does not exceed 1.0 by much
    x_data_example = np.array([0, 5, 10, 20, 40, 80, 160, 260, 360])
    y_data_example = np.array([0.70, 0.72, 0.80, 0.95, 1.00, 1.00, 1.00, 1.00, 1.00])
    
    # Fit the logistic model
    params, cov = fit_logistic_model(x_data_example, y_data_example)
    L_fit, k_fit, x0_fit, offset_fit = params
    
    
    # ---------------------------
    # Get parameters for the Gaussian tail model
    # ---------------------------
    # Example data
    x_data_example = np.array([0, 5, 10, 20, 40, 80, 160, 260, 360])
    y_data_example = np.array([0.70, 0.72, 0.80, 0.95, 1.05, 1.02, 1.00, 1.00, 1.00])
    
    # Fit and plot
    fit_params, fit_cov = fit_gauss_tail(x_data_example, y_data_example)
    Y_normal_fit, a_fit, b_fit, c_fit, p_fit, q_fit = fit_params
    

    
    
    