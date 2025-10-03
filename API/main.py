#main.py

from fastapi import FastAPI, HTTPException, Query
# CRITICAL: Import CORS Middleware
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel, Field
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import numpy as np
import pandas as pd

# Import your processing functions
from air_quality_processor import (
    get_tempo_heatmap_data,
    get_tempo_averaged_data,
    calculate_aqi,
    correlate_tempo_airnow,
    interpolate_heatmap,
    get_cities_in_bbox,
    get_spatially_averaged_timeseries, # Import the new function
    TEMPO_VARIABLE_MAP,
    TEMPO_COLUMN_MAP,
    get_airnow_variables
)

# Initialize FastAPI app
app = FastAPI(
    title="Air Quality Data API",
    description="API for fetching, processing, and analyzing TEMPO and AirNow air quality data.",
    version="1.0.0"
)

# --- CRITICAL: Add CORS Middleware ---
# This must be added immediately after app initialization.
# It fixes the "405 Method Not Allowed" on OPTIONS requests.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (like your local html file)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- Pydantic Models for API ---

class BoundingBox(BaseModel):
    min_lon: float = Field(..., description="Minimum longitude (e.g., -125.0)")
    min_lat: float = Field(..., description="Minimum latitude (e.g., 24.0)")
    max_lon: float = Field(..., description="Maximum longitude (e.g., -66.9)")
    max_lat: float = Field(..., description="Maximum latitude (e.g., 49.5)")

    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)

class InterpolatedHeatmapResponse(BaseModel):
    lon_grid: List[List[float]]
    lat_grid: List[List[float]]
    values: List[List[Optional[float]]]
    unit: str

class AveragedDataPoint(BaseModel):
    time: datetime
    value: float

class CorrelationResult(BaseModel):
    pearson: Optional[float]
    spearman: Optional[float]
    kendall: Optional[float]

class AQIResult(BaseModel):
    aqi: Optional[int]
    category: str

class CityData(BaseModel):
    name: str
    lon: float
    lat: float

# --- API Endpoints ---

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Air Quality Data API. CORS is enabled."}


@app.post("/spatially_averaged_timeseries", response_model=List[AveragedDataPoint])
async def get_spatially_averaged_timeseries_data(
    bbox: BoundingBox,
    day: str = Query(..., description="The day for the timeseries, e.g., '2025-10-02'"),
    product: str = Query(..., description="The data product, e.g., 'AirQuality.airnow.ozone'")
):
    """
    Generates a spatially averaged time series for a given day, bounding box, and product.
    This is ideal for plotting time series graphs.
    """
    try:
        timeseries = get_spatially_averaged_timeseries(
            day=day,
            bbox=bbox.to_tuple(),
            product=product
        )

        if timeseries is None or timeseries.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No time series data found for product '{product}' on {day} in the specified bounding box."
            )

        # Format the response into a list of data points, excluding any NaN values
        response_data = [
            AveragedDataPoint(time=index.to_pydatetime(), value=float(value))
            for index, value in timeseries.items() if pd.notna(value)
        ]
        
        return response_data

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Data processing error: {e}")
    except Exception as e:
        # Log the error for debugging purposes
        print(f"Server error in /spatially_averaged_timeseries: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")


@app.post("/heatmap_data", response_model=InterpolatedHeatmapResponse)
async def get_interpolated_heatmap_data(
    bbox: BoundingBox,
    species: str = Query(..., description="Species to fetch (O3, NO2, HCHO)", regex="^(O3|NO2|HCHO)$"),
    start_date: str = Query(..., description="Start date (e.g., '2025-09-29 00')"),
    end_date: str = Query(..., description="End date (e.g., '2025-09-30 23:59:59')"),
    lat_bins: int = Query(40, ge=10, le=100, description="Number of latitude bins for initial grid"),
    lon_bins: int = Query(50, ge=10, le=100, description="Number of longitude bins for initial grid"),
    resolution_factor: int = Query(10, ge=1, le=20, description="Factor to increase grid resolution for interpolation")
):
    """
    Fetches TEMPO data, grids it, and interpolates for heatmap overlay.
    """
    # Note: This process can take time depending on the bbox size and pyrsig speed.
    try:
        coarse_data = get_tempo_heatmap_data(
            species=species,
            start_date=start_date,
            end_date=end_date,
            bbox=bbox.to_tuple(),
            lat_bins=lat_bins,
            lon_bins=lon_bins
        )

        if coarse_data is None:
            raise HTTPException(status_code=404, detail=f"No {species} data found for these parameters.")

        grid_lon, grid_lat, smooth_values = interpolate_heatmap(coarse_data, resolution_factor=resolution_factor)

        if grid_lon.size == 0:
             raise HTTPException(status_code=404, detail="Data found, but could not be interpolated.")

        return InterpolatedHeatmapResponse(
            lon_grid=grid_lon.tolist(),
            lat_grid=grid_lat.tolist(),
            values=[[None if np.isnan(val) else float(val) for val in row] for row in smooth_values.tolist()],
            unit=TEMPO_COLUMN_MAP[species.upper()]
        )
    except RuntimeError as e:
        print(f"Server Error in /heatmap_data: {e}") # Print to server logs for debugging
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/averaged_data", response_model=List[AveragedDataPoint])
async def get_time_averaged_data(
    bbox: BoundingBox,
    species: str = Query(..., description="Species to fetch (O3, NO2, HCHO)"),
    start_date: str = Query(...),
    end_date: str = Query(...),
    freq: str = Query('1H')
):
    # Note: This is a POST request. Sending a GET will result in 405 Method Not Allowed.
    try:
        averaged_series = get_tempo_averaged_data(
            species=species,
            start_date=start_date,
            end_date=end_date,
            bbox=bbox.to_tuple(),
            freq=freq
        )
        if averaged_series is None or averaged_series.empty:
            raise HTTPException(status_code=404, detail="No averaged data found.")

        return [
            AveragedDataPoint(time=idx.to_pydatetime(), value=float(val))
            for idx, val in averaged_series.items() if not pd.isna(val)
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/aqi", response_model=AQIResult)
async def get_aqi_value(
    pollutant: str = Query(...),
    concentration: float = Query(..., gt=0)
):
    aqi, category = calculate_aqi(pollutant, concentration)
    if aqi is None:
        raise HTTPException(status_code=400, detail=category)
    return AQIResult(aqi=aqi, category=category)
