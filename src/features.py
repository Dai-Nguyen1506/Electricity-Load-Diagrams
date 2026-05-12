"""
Feature engineering for time series forecasting.
"""
import numpy as np
import holidays

def create_static_features(df):
    df_features = df.copy()
    
    # 2. Calendar Features
    df_features['hour'] = df_features.index.hour
    df_features['month'] = df_features.index.month

    df_features['day_of_week'] = df_features.index.dayofweek
    df_features['day_of_month'] = df_features.index.day
    df_features['day_of_year'] = df_features.index.dayofyear

    # 3. Cyclical Features (Sin/Cos)
    # Daily cycle (24h)
    df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
    df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)

    # Annual cycle (365.25 days) - Crucial replacement for yearly lag
    df_features['year_sin'] = np.sin(2 * np.pi * (df_features['day_of_year']-1) / 365.25)
    df_features['year_cos'] = np.cos(2 * np.pi * (df_features['day_of_year']-1) / 365.25)

    return df_features


def create_holidays_features(df):
    df_holidays = df.copy()
    
    # 1. Initialize calendar
    # Fetch holidays from 2011 to 2014
    pt_holidays = holidays.Portugal(years=range(2011, 2014))

    # 2. Create holiday name column (using original names from the library)
    # If not a holiday, set to "None"
    df_holidays['holiday_name'] = df_holidays.index.normalize().map(lambda d: pt_holidays.get(d) if d in pt_holidays else "None")

    # 3. Convert holiday names to numeric codes (Label Encoding)
    # Same holidays across different years will share the same code
    df_holidays['holiday_name'] = df_holidays['holiday_name'].astype('category')

    return df_holidays