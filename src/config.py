"""
Global configuration for the project.
"""

FREQUENCY = "H"

SEASONALITY = {
    "daily": 24,
    "weekly": 168,
    "yearly": 8760
}

FORECAST_HORIZONS = [1, 24, 168]
RANDOM_SEED = 42
