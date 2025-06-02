import numpy as np

region_tree_inputs = {
    "CWW": {
        # area of trees themselves (ha) for LMU0 to LMU8
        "estimated_area_trees_l": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
        # tree fertilizer/soil scalar for LMU0 to LMU8
        "tree_fert_soil_scalar": np.array([0.35, 0.83, 1, 0.78, 0.82, 0.82, 0.89, 0.90, 0.7]), #average of crop lmu scalars
        #growth scalar for region
        "regional_growth_scalar": 1,
        # tree growth scalar for LMU0 to LMU8
        "lmu_growth_scalar_l": np.array([0.40, 0.74, 1, 0.82, 0.86, 0.94, 0.86, 0.88, 0.7]),
        "lmu_carbon_scalar_l": np.array([0.70, 0.87, 1.00, 0.91, 0.93, 0.97, 0.93, 0.94, 0.85]), #this is less because as the trees on the good soils reach mature size the gap will cloase between soil types.
    },
    "EWW": {
        "estimated_area_trees_l": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
        # Use np.nan for missing values (None converted to np.nan)
        "tree_fert_soil_scalar": np.array([0.69, 0.98, 1.02, 1, 0.96, 0.81, 0.81, 0.56, np.nan]),
        #growth scalar for region
        "regional_growth_scalar": 0.64,
        # tree growth scalar for LMU0 to LMU8
        "lmu_growth_scalar_l": np.array([0.69, 0.98, 1.02, 1, 0.96, 0.81, 0.81, 0.56, np.nan]),
        "lmu_carbon_scalar_l": np.array([0.845, 0.99, 1.01, 1.0, 0.98, 0.905, 0.905, 0.78, np.nan]), #this is less because as the trees on the good soils reach mature size the gap will cloase between soil types.
    },
    "GSW": {
        "estimated_area_trees_l": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
        "tree_fert_soil_scalar": np.array([0.62, 0.62, 0.77, 1, 1, np.nan, np.nan, np.nan, np.nan]),
        #growth scalar for region
        "regional_growth_scalar": 1.64,
        # tree growth scalar for LMU0 to LMU8
        "lmu_growth_scalar_l": np.array([0.62, 0.62, 0.77, 1, 1, np.nan, np.nan, np.nan, np.nan]),
        "lmu_carbon_scalar_l": np.array([0.81, 0.81, 0.885, 1.0, 1.0, np.nan, np.nan, np.nan, np.nan]), #this is less because as the trees on the good soils reach mature size the gap will cloase between soil types.
    },
    "SWV": {
        "estimated_area_trees_l": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
        "tree_fert_soil_scalar": np.array([1, 1, 1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]),
        #growth scalar for region
        "regional_growth_scalar": 1.97,
        # tree growth scalar for LMU0 to LMU8
        "lmu_growth_scalar_l": np.array([1, 1, 1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]),
        "lmu_carbon_scalar_l": np.array([1, 1, 1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]),
    }
}
