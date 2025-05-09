import numpy as np

region_tree_inputs = {
    "CWW": {
        # area of trees themselves (ha) for LMU0 to LMU8
        "estimated_area_trees_l": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
        # tree fertilizer/soil scalar for LMU0 to LMU8
        "tree_fert_soil_scalar": np.array([0.35, 0.83, 1, 0.78, 0.82, 0.82, 0.89, 0.90, 0.65]),
        #growth scalar for region
        "regional_growth_scalar": 1,
        # tree growth scalar for LMU0 to LMU8
        "lmu_growth_scalar_l": np.array([0.35, 0.83, 1, 0.78, 0.82, 0.82, 0.89, 0.90, 0.65]),
    },
    "EWW": {
        "estimated_area_trees_l": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
        # Use np.nan for missing values (None converted to np.nan)
        "tree_fert_soil_scalar": np.array([0.69, 0.98, 1.02, 1, 0.96, 0.81, 0.81, 0.56, np.nan]),
        #growth scalar for region
        "regional_growth_scalar": 1,
        # tree growth scalar for LMU0 to LMU8
        "lmu_growth_scalar_l": np.array([0.69, 0.98, 1.02, 1, 0.96, 0.81, 0.81, 0.56, np.nan]),
    },
    "GSW": {
        "estimated_area_trees_l": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
        "tree_fert_soil_scalar": np.array([0.62, 0.62, 0.77, 1, 1, np.nan, np.nan, np.nan, np.nan]),
        #growth scalar for region
        "regional_growth_scalar": 1,
        # tree growth scalar for LMU0 to LMU8
        "lmu_growth_scalar_l": np.array([0.62, 0.62, 0.77, 1, 1, np.nan, np.nan, np.nan, np.nan]),
    },
    "SWV": {
        "estimated_area_trees_l": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
        "tree_fert_soil_scalar": np.array([1, 1, 1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]),
        #growth scalar for region
        "regional_growth_scalar": 1,
        # tree growth scalar for LMU0 to LMU8
        "lmu_growth_scalar_l": np.array([1, 1, 1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]),
    }
}
