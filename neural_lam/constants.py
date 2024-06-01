# Third-party
import cartopy
import numpy as np

WANDB_PROJECT = "seacast"

SECONDS_IN_YEAR = (
    365 * 24 * 60 * 60
)  # Assuming no leap years in dataset (2024 is next)

# Log prediction error for these lead times
VAL_STEP_LOG_ERRORS = np.array([1, 3, 4])

# Log these metrics to wandb as scalar values for
# specific variables and lead times
# List of metrics to watch, including any prefix (e.g. val_rmse)
METRICS_WATCH = []

# Dict with variables and lead times to log watched metrics for
# Format is a dictionary that maps from a variable index to
# a list of lead time steps
VAR_LEADS_METRICS_WATCH = {}

# Variable names
PARAM_NAMES = [
    "Eastward sea water velocity",
    "Northward sea water velocity",
    "Ocean mixed layer thickness defined by sigma theta",
    "Sea water salinity",
    "Sea surface height above geoid",
    "Sea water potential temperature",
    "Sea water potential temperature at sea floor",
]

PARAM_NAMES_SHORT = [
    "uo",
    "vo",
    "mlotst",
    "so",
    "zos",
    "thetao",
    "bottomT",
]

PARAM_UNITS = [
    "m/s",
    "m/s",
    "-",
    "‰",
    "m",
    "°C",
    "°C",
]

PARAM_COLORMAPS = [
    "coolwarm",
    "coolwarm",
    "viridis",
    "viridis",
    "viridis",
    "viridis",
    "viridis",
]

LEVELS = [
    True,
    True,
    False,
    True,
    False,
    True,
    False,
]

# Projection and grid
GRID_SHAPE = (371, 1013)  # (y, x)
GRID_LIMITS = [-6, 36.291668, 30.1875, 45.979168]
PROJECTION = cartopy.crs.PlateCarree()

# Data dimensions
GRID_FORCING_DIM = 8 * 3  # 2 feat. for 3 time-step window + 0 batch-static
GRID_STATE_DIM = 75

DEPTHS = [
    1.0182366,
    5.4649634,
    10.536604,
    16.270586,
    22.706392,
    29.885643,
    37.852192,
    46.652210,
    56.334286,
    66.949490,
    78.551500,
    91.196630,
    104.94398,
    119.85543,
    135.99580,
    153.43285,
    172.23735,
    192.48314,
]


# New lists
EXP_PARAM_NAMES_SHORT = []
EXP_PARAM_UNITS = []
EXP_PARAM_COLORMAPS = []

for name, unit, colormap, levels_applies in zip(
    PARAM_NAMES_SHORT, PARAM_UNITS, PARAM_COLORMAPS, LEVELS
):
    if levels_applies:
        for depth in DEPTHS:
            depth_int = round(depth)
            EXP_PARAM_NAMES_SHORT.append(f"{name}_{depth_int}")
            EXP_PARAM_UNITS.append(unit)
            EXP_PARAM_COLORMAPS.append(colormap)
    else:
        EXP_PARAM_NAMES_SHORT.append(name)
        EXP_PARAM_UNITS.append(unit)
        EXP_PARAM_COLORMAPS.append(colormap)
