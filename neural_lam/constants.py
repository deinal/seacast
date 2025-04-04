# Third-party
import cartopy
import cmocean
import numpy as np

WANDB_PROJECT = "seacast"

# Log prediction error for these lead times
VAL_STEP_LOG_ERRORS = np.array([1, 2, 3, 4])
TEST_STEP_LOG_ERRORS = np.array(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
)

# Sample lengths
SAMPLE_LEN = {
    "train": 6,
    "val": 6,
    "test": 17,
}

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
    "Sea water salinity",
    "Sea water potential temperature",
    "Sea surface height above geoid",
]

PARAM_NAMES_SHORT = [
    "uo",
    "vo",
    "so",
    "thetao",
    "zos",
]

PLOT_NAMES = [
    "Zonal current",
    "Meridional current",
    "Salinity",
    "Temperature",
    "Sea surface height",
]

PARAM_UNITS = [
    "m/s",
    "m/s",
    "psu",
    "°C",
    "m",
]

PARAM_COLORMAPS = [
    cmocean.cm.balance,
    cmocean.cm.balance,
    cmocean.cm.haline,
    cmocean.cm.thermal,
    cmocean.cm.curl,
]

DIVERGING = [
    True,
    True,
    False,
    False,
    True,
]

LEVELS = [
    True,
    True,
    True,
    True,
    False,
]

# Projection and grid
GRID_SHAPE = (371, 1013)  # (y, x)
GRID_LIMITS = [-6, 36.291668, 30.1875, 45.979168]
PROJECTION = cartopy.crs.PlateCarree()

# Data dimensions
GRID_FORCING_DIM = 6 * 3  # 6 feat. for 3 time-step window + 0 batch-static
GRID_STATE_DIM = 73

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

ATM_FORCING = ["tau_u", "tau_v", "t2m", "msl"]

# New lists
EXP_PARAM_NAMES = []
EXP_PARAM_NAMES_SHORT = []
EXP_PARAM_UNITS = []
EXP_PARAM_COLORMAPS = []
EXP_DIVERGING = []

for name, short_name, unit, colormap, diverging, levels_applies in zip(
    PARAM_NAMES,
    PARAM_NAMES_SHORT,
    PARAM_UNITS,
    PARAM_COLORMAPS,
    DIVERGING,
    LEVELS,
):
    if levels_applies:
        for depth in DEPTHS:
            depth_int = round(depth)
            EXP_PARAM_NAMES.append(f"{name}_{depth_int}")
            EXP_PARAM_NAMES_SHORT.append(f"{short_name}_{depth_int}")
            EXP_PARAM_UNITS.append(unit)
            EXP_PARAM_COLORMAPS.append(colormap)
            EXP_DIVERGING.append(diverging)
    else:
        EXP_PARAM_NAMES.append(name)
        EXP_PARAM_NAMES_SHORT.append(short_name)
        EXP_PARAM_UNITS.append(unit)
        EXP_PARAM_COLORMAPS.append(colormap)
        EXP_DIVERGING.append(diverging)

PLOT_NAMES_MAP = dict(zip(PARAM_NAMES_SHORT, PLOT_NAMES))
