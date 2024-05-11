# Third-party
import cartopy
import numpy as np

WANDB_PROJECT = "balticnet"

SECONDS_IN_YEAR = (
    365 * 24 * 60 * 60
)  # Assuming no leap years in dataset (2024 is next)

# Log prediction error for these lead times
VAL_STEP_LOG_ERRORS = np.array([1, 3, 6])

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
    "Sea water potential temperature",
    "Sea water potential temperature at sea floor",
    "Ocean mixed layer thickness defined by sigma theta",
    "Sea ice area fraction",
    "Sea ice thickness",
    "Sea water salinity",
    "Sea water salinity at sea floor",
    "Eastward sea water velocity",
    "Northward sea water velocity",
    "Mass concentration of chlorophyll a in sea water",
    "Mole concentration of ammonium in sea water",
    "Mole concentration of nitrate in sea water",
    "Mole concentration of dissolved molecular oxygen in sea water",
    "Mole concentration of dissolved molecular oxygen at the bottom",
    "Sea water pH reported on total scale",
    "Mole concentration of phosphate in sea water",
    "Surface partial pressure of carbon dioxide in sea water",
    "Secchi depth of sea water",
]

PARAM_NAMES_SHORT = [
    "thetao",
    "bottomT",
    "mlotst",
    "siconc",
    "sithick",
    "so",
    "sob",
    "uo",
    "vo",
    "chl",
    "nh4",
    "no3",
    "o2",
    "o2b",
    "ph",
    "po4",
    "spco2",
    "zsd",
]

PARAM_UNITS = [
    "°C",
    "°C",
    "m",
    "-",
    "m",
    "‰",
    "‰",
    "m/s",
    "m/s",
    "mg/m³",
    "mmol/m³",
    "mmol/m³",
    "mmol/m³",
    "mmol/m³",
    "-",
    "mmol/m³",
    "Pa",
    "m",
]

PARAM_COLORMAPS = [
    "viridis",
    "viridis",
    "viridis",
    "viridis",
    "viridis",
    "viridis",
    "viridis",
    "coolwarm",
    "coolwarm",
    "viridis",
    "viridis",
    "viridis",
    "viridis",
    "viridis",
    "viridis",
    "viridis",
    "viridis",
    "viridis",
]

# Projection and grid
GRID_SHAPE = (774, 763)  # (y, x)
GRID_LIMITS = [9.04, 30.21, 53.01, 65.89]
PROJECTION = cartopy.crs.PlateCarree()

# Data dimensions
GRID_FORCING_DIM = 2 * 3  # 2 feat. for 3 time-step window + 0 batch-static
GRID_STATE_DIM = 18
