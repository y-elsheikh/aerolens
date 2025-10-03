# air_quality_processor.py

import pandas as pd
import numpy as np
import xarray as xr
import pyrsig
import os
import geonamescache
from scipy.interpolate import griddata
from typing import List, Tuple, Dict, Optional

# Ensure the pycno cache directory exists
os.makedirs(os.path.expanduser('~/.pycno'), exist_ok=True)

bbox2 = (-125, 24, -66.9, 49.5)

#TEMPO_VARIABLE_MAP = {
#    'O3': 'tempo.l3.o3tot.column_amount_o3',
#    # Use the new, potentially smaller, NO2 variable
#    'NO2': 'tempo.l3.no2.vertical_column_total',
#    'HCHO': 'tempo.l3.hcho.vertical_column' # Assuming HCHO might also be large
#}

# --- Constants for TEMPO Variables ---
TEMPO_VARIABLE_MAP = {"CO": "tropomi.nrti.co.carbonmonoxide_total_column",
                      'O3': "modis.mod7.Total_Ozone",
                      "NO2": "tropomi.nrti.no2.nitrogendioxide_tropospheric_column",
                      "HCHO": "tropomi.nrti.hcho.formaldehyde_tropospheric_vertical_column"}

TEMPO_COLUMN_MAP = {
    'O3': 'Total_Ozone(Dobson)',
    # Update the column name to match the new variable
    'NO2': 'nitrogendioxide_tropospheric_column(molecules/cm2)',
    "CO": "carbonmonoxide_total_column(molecules/cm2)",
    'HCHO': 'formaldehyde_tropospheric_vertical_column(molecules/cm2)'
}

#TEMPO_COLUMN_MAP = {
#    'O3': 'o3_column_amount_o3(DU)',
#    # Update the column name to match the new variable
#    'NO2': 'no2_vertical_column_total(molecules/cm2)',
#    'HCHO': 'vertical_column(molecules/cm2)'
#}

# --- TEMPO Data Functions ---
LOCAL_FILE_MAP = {
    "NO2": "tropomi.nrti.no2.nitrogendioxide_tropospheric_column_2025-10-01T110000Z_2025-10-03T110000Z.csv",
    "HCHO": "tropomi.nrti.hcho.formaldehyde_tropospheric_vertical_column_2025-10-02T000000Z_2025-10-02T235959Z.csv",
    "O3": "modis.mod7.Total_Ozone_2025-10-02T000000Z_2025-10-02T235959Z.csv"
}

TEMPO_COLUMN_MAP = {
    "O3": "Total_Ozone(Dobson)",
    "NO2": "nitrogendioxide_tropospheric_column(molecules/cm2)",
    "HCHO": "formaldehyde_tropospheric_vertical_column(molecules/cm2)"
}


def get_spatially_averaged_timeseries(day: str, bbox: Tuple[float, float, float, float], product: str) -> Optional[pd.Series]:
    """
    Fetches and creates a spatially averaged time series for a given day, bbox, and product.

    Args:
        day (str): The day for the query (e.g., '2025-10-02').
        bbox (Tuple[float, float, float, float]): The bounding box for the spatial query.
        product (str): The product to query (e.g., 'AirQuality.airnow.no2').

    Returns:
        Optional[pd.Series]: A pandas Series with the time series data, or None if no data is found.
    """
    start_date = f"2025-09-29 00:00:00"
    end_date = f"2025-10-01 23:59:59"

    try:
        rsig_api = pyrsig.RsigApi(bdate=start_date, edate=end_date, bbox=bbox, overwrite=True)
        
        # Define the column name based on the product
        if product == 'NO2':
            product = "tropomi.nrti.no2.nitrogendioxide_tropospheric_column"
            column_name = 'nitrogendioxide_tropospheric_column(molecules/cm2)',
        elif product == 'O3':
            product = "modis.mod7.Total_Ozone"
            column_name = 'Total_Ozone(Dobson)'
        elif product == 'HCHO':
            product = "tropomi.nrti.hcho.formaldehyde_tropospheric_vertical_column"
            column_name = 'formaldehyde_tropospheric_vertical_column(molecules/cm2)'
        else:
            # Fallback for other potential products
            # This might need adjustment if other products are used
            column_name = product.split('.')[-1]
        
        df = rsig_api.to_dataframe(product, parse_dates=True, unit_keys=True)
        print(df.columns)
        if df.empty:
            return None

        # Ensure the time column is in datetime format
        if 'Timestamp(UTC)' in df.columns:
            df['time'] = pd.to_datetime(df['Timestamp(UTC)'], utc=True)
            df.set_index('time', inplace=True)
        
        # Spatially average the data by resampling time
        # The mean will be calculated for all data points within each time interval
        if True:
            print("yay!!!!!")
            print(df.columns[3])
            # Resample to a regular interval (e.g., 1 hour) and take the mean
            return df[df.columns[3]].resample("1min").mean()
        else:
            raise KeyError(f"Column '{column_name}' not found in the dataframe for product '{product}'.")

    except Exception as e:
        # In a real application, you might want to log this error
        print(f"An error occurred while fetching spatially averaged data: {e}")
        return None


def _load_species_df(species: str) -> pd.DataFrame:
    """
    Internal helper to load the CSV for a given species.
    """
    if species.upper() not in LOCAL_FILE_MAP:
        raise ValueError(f"Species not supported: {species}")
    file_path = LOCAL_FILE_MAP[species.upper()]
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Local file not found: {file_path}")

    # Read CSV
    df = pd.read_csv(file_path, delimiter="\t")

    print(df)

    # Normalize timestamp column name
    if "Timestamp(UTC)" in df.columns:
        df = df.rename(columns={"Timestamp(UTC)": "time"})
    elif "time" not in df.columns:
        raise KeyError("CSV file must contain either 'Timestamp(UTC)' or 'time' column")

    # Convert to datetime
    df["time"] = pd.to_datetime(df["time"], utc=True)

    return df


def get_tempo_heatmap_data(species: str, start_date: str, end_date: str,
                           bbox: Tuple[float, float, float, float],
                           lat_bins: int = 10, lon_bins: int = 10) -> Optional[xr.DataArray]:
    """
    Load species CSV and bin into a 2D grid for heatmap.
    Ignores start_date and end_date.
    """
    df = _load_species_df(species)
    column_name = TEMPO_COLUMN_MAP[species.upper()]

    if df.empty:
        return None

    # Create grid edges from bbox
    latedges = np.linspace(bbox[1], bbox[3], lat_bins + 1)
    lonedges = np.linspace(bbox[0], bbox[2], lon_bins + 1)

    latbin = pd.cut(df['LATITUDE(deg)'], latedges, labels=(latedges[:-1] + latedges[1:]) / 2).astype('f')
    lonbin = pd.cut(df['LONGITUDE(deg)'], lonedges, labels=(lonedges[:-1] + lonedges[1:]) / 2).astype('f')

    ds = df.groupby([
        pd.Grouper(key='time', freq='1h'), latbin, lonbin
    ]).mean(numeric_only=True)[[column_name]].to_xarray()

    return ds[column_name].mean('time')

def get_tempo_averaged_data(species: str, start_date: str, end_date: str, bbox: Tuple[float, float, float, float], freq: str = '1H') -> Optional[pd.Series]:
    """
    Fetches and returns time-averaged TEMPO data for a specific time and location.

    Args:
        species (str): The species to fetch ('O3', 'NO2', or 'HCHO').
        start_date (str): The start date for the query.
        end_date (str): The end date for the query.
        bbox (tuple): Bounding box for the query.
        freq (str): The pandas frequency string for grouping (e.g., '1H' for hourly).

    Returns:
        pd.Series: A time-series of averaged data, or None on error.
    """
    if species.upper() not in TEMPO_VARIABLE_MAP:
        raise ValueError("Species not supported. Choose from 'O3', 'NO2', 'HCHO'.")

    variable_key = TEMPO_VARIABLE_MAP[species.upper()]
    column_name = TEMPO_COLUMN_MAP[species.upper()]

    try:
        rsig_api = pyrsig.RsigApi(bdate=start_date, edate=end_date, bbox=bbox)
        df = rsig_api.to_dataframe(variable_key, parse_dates=True, unit_keys=True)
        if df.empty:
            # print(f"No TEMPO data found for {species} in the given time/location.") # Removed print for API
            return None

        df['time'] = pd.to_datetime(df['time'], utc=True)
        averaged_series = df.groupby(pd.Grouper(key='time', freq=freq)).mean(numeric_only=True)[column_name]
        return averaged_series
    except Exception as e:
        # print(f"Error fetching averaged TEMPO data for {species}: {e}") # Removed print for API
        raise RuntimeError(f"Error fetching averaged TEMPO data: {e}") from e

# --- AirNow and Correlation Functions ---

def get_airnow_variables() -> List[str]:
    """
    Lists all available variable keys from the AirNow dataset.
    Initializes a temporary API object to fetch keys.
    """
    try:
        # A minimal API object is needed just to access the keys
        temp_api = pyrsig.RsigApi(bdate='2024-01-01 00', edate='2024-01-01 01', bbox=(-1, -1, 1, 1))
        return [k for k in temp_api.keys() if 'airnow' in k]
    except Exception as e:
        # print(f"Could not fetch AirNow variables: {e}") # Removed print for API
        raise RuntimeError(f"Error fetching AirNow variables: {e}") from e

def calculate_aqi(pollutant: str, concentration: float) -> Tuple[Optional[int], str]:
    """
    Calculates the Air Quality Index (AQI) for a given pollutant and concentration.
    """
    # EPA AQI Breakpoints
    breakpoints = {
        'O3_8hr': [(0, 50, 0, 54), (51, 100, 55, 70), (101, 150, 71, 85), (151, 200, 86, 105), (201, 300, 106, 200)],
        'PM25': [(0, 50, 0.0, 12.0), (51, 100, 12.1, 35.4), (101, 150, 35.5, 55.4), (151, 200, 55.5, 150.4), (201, 300, 150.5, 250.4)],
        'CO': [(0, 50, 0.0, 4.4), (51, 100, 4.5, 9.4), (101, 150, 9.5, 12.4), (151, 200, 12.5, 15.4), (201, 300, 15.5, 30.4)],
        'SO2_1hr': [(0, 50, 0, 35), (51, 100, 36, 75), (101, 150, 76, 185), (151, 200, 186, 304)],
        'NO2': [(0, 50, 0, 53), (51, 100, 54, 100), (101, 150, 101, 360), (151, 200, 361, 649)]
    }
    categories = {(0, 50): "Good", (51, 100): "Moderate", (101, 150): "Unhealthy for Sensitive Groups", (151, 200): "Unhealthy", (201, 300): "Very Unhealthy", (301, 500): "Hazardous"}
    pollutant_map = {'O3': 'O3_8hr', 'PM25': 'PM25', 'CO': 'CO', 'SO2': 'SO2_1hr', 'NO2': 'NO2'}
    key = pollutant_map.get(pollutant.upper())
    if not key: return None, "Pollutant not supported"
    for I_low, I_high, C_low, C_high in breakpoints[key]:
        if C_low <= concentration <= C_high:
            aqi = round(((I_high - I_low) / (C_high - C_low)) * (concentration - C_low) + I_low)
            for (start, end), cat in categories.items():
                if start <= aqi <= end: return aqi, cat
    if concentration > breakpoints[key][-1][3]: return 500, "Hazardous"
    return None, "Concentration out of range"


def correlate_tempo_airnow(start_date: str, end_date: str, bbox: Tuple[float, float, float, float], tempo_species: str = 'O3', airnow_species: str = 'ozone', freq: str = '1H') -> Optional[Dict[str, float]]:
    """
    Fetches, aligns, and correlates TEMPO and AirNow data.

    Returns:
        dict: A dictionary of correlation coefficients, or None on error.
    """
    try:
        rsig_api = pyrsig.RsigApi(bdate=start_date, edate=end_date, bbox=bbox)

        # --- Get TEMPO data ---
        tempo_var_key = TEMPO_VARIABLE_MAP[tempo_species.upper()]
        tempo_col_name = TEMPO_COLUMN_MAP[tempo_species.upper()]
        tempo_df = rsig_api.to_dataframe(tempo_var_key, parse_dates=True, unit_keys=True)
        if tempo_df.empty:
            # print(f"No TEMPO data found for {tempo_species}.") # Removed print for API
            return None
        tempo_df['time'] = pd.to_datetime(tempo_df['time'], utc=True)
        tempo_series = tempo_df.groupby(pd.Grouper(key='time', freq=freq)).mean(numeric_only=True)[tempo_col_name]

        # --- Get AirNow data ---
        airnow_var_key = f'airnow.{airnow_species.lower()}'
        airnow_df = rsig_api.to_dataframe(airnow_var_key, parse_dates=True, unit_keys=True)
        if airnow_df.empty:
            # print(f"No AirNow data found for {airnow_species}.") # Removed print for API
            return None

        airnow_df['time'] = pd.to_datetime(airnow_df['time'], utc=True)
        airnow_df.set_index('time', inplace=True)

        airnow_col_name = f'{airnow_species.lower()}(ppb)'
        if airnow_col_name not in airnow_df.columns:
            airnow_col_name = airnow_species.lower()
            if airnow_col_name not in airnow_df.columns:
                raise KeyError(f"Could not find a column for '{airnow_species}' in the AirNow data.")

        airnow_series = airnow_df.groupby(pd.Grouper(freq=freq)).mean(numeric_only=True)[airnow_col_name]

        # --- Alignment and Correlation ---
        common_idx = airnow_series.index.intersection(tempo_series.index.floor(freq))
        airnow_aligned = airnow_series.loc[common_idx].dropna()
        tempo_aligned = tempo_series.reindex(common_idx, method='nearest').dropna()

        final_common_idx = airnow_aligned.index.intersection(tempo_aligned.index)
        airnow_aligned = airnow_aligned.loc[final_common_idx]
        tempo_aligned = tempo_aligned.loc[final_common_idx]

        if airnow_aligned.empty or tempo_aligned.empty:
            # print("No overlapping data found after alignment and cleaning.") # Removed print for API
            return None

        return {
            'pearson': airnow_aligned.corr(tempo_aligned, method='pearson'),
            'spearman': airnow_aligned.corr(tempo_aligned, method='spearman'),
            'kendall': airnow_aligned.corr(tempo_aligned, method='kendall')
        }
    except KeyError as ke:
        # print(f"Data column not found: {ke}. Please check the 'airnow_species' name.") # Removed print for API
        raise RuntimeError(f"Data column not found for AirNow species: {ke}") from ke
    except Exception as e:
        # print(f"Error during correlation: {e}") # Removed print for API
        raise RuntimeError(f"Error during correlation: {e}") from e


def interpolate_heatmap(data_array: xr.DataArray, resolution_factor: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolates an xarray.DataArray to a finer grid for a smoother heatmap.

    Args:
        data_array (xr.DataArray): The coarse, binned heatmap data.
        resolution_factor (int): Factor by which to increase the grid resolution.

    Returns:
        tuple: (grid_lon, grid_lat, grid_z) - The new longitude grid, latitude grid,
               and interpolated data grid, ready for plotting.
    """
    if data_array is None:
        return np.array([]), np.array([]), np.array([])

    lon_coords = data_array['LONGITUDE(deg)'].values
    lat_coords = data_array['LATITUDE(deg)'].values

    if len(lon_coords) < 2 or len(lat_coords) < 2:
        # Not enough points to interpolate
        return np.array([]), np.array([]), np.array([])

    lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)

    lon_flat = lon_grid.flatten()
    lat_flat = lat_grid.flatten()
    values_flat = data_array.values.flatten()

    valid_mask = ~np.isnan(values_flat)

    if not np.any(valid_mask):
        # print("Warning: All data points are NaN. Cannot interpolate.") # Removed print for API
        return np.array([]), np.array([]), np.array([])

    num_lon_fine = len(lon_coords) * resolution_factor
    num_lat_fine = len(lat_coords) * resolution_factor

    grid_lon_fine, grid_lat_fine = np.meshgrid(
        np.linspace(lon_coords.min(), lon_coords.max(), num_lon_fine),
        np.linspace(lat_coords.min(), lat_coords.max(), num_lat_fine)
    )

    grid_z = griddata(
        (lon_flat[valid_mask], lat_flat[valid_mask]),
        values_flat[valid_mask],
        (grid_lon_fine, grid_lat_fine),
        method='cubic'
    )

    return grid_lon_fine, grid_lat_fine, grid_z


def get_cities_in_bbox(bbox: Tuple[float, float, float, float], min_population: int = 100000) -> Dict[str, Tuple[float, float]]:
    """
    Gets a list of major cities within a given bounding box using geonamescache.

    Args:
        bbox (tuple): The map bounding box (min_lon, min_lat, max_lon, max_lat).
        min_population (int): The minimum population for a city to be included.

    Returns:
        dict: A dictionary of city names and their (longitude, latitude) coordinates.
    """
    gc = geonamescache.GeonamesCache()
    cities = gc.get_cities()

    cities_in_view = {}
    min_lon, min_lat, max_lon, max_lat = bbox # Corrected order for consistency

    for city_id, city_data in cities.items():
        lat = city_data.get('latitude')
        lon = city_data.get('longitude')
        pop = city_data.get('population')
        name = city_data.get('name')

        if (min_lon <= lon <= max_lon and
            min_lat <= lat <= max_lat and
            pop >= min_population):

            cities_in_view[name] = (lon, lat)

    return cities_in_view
