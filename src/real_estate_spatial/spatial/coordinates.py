"""Geospatial coordinate calculations and geometric utilities."""

from __future__ import annotations

import math
from typing import Sequence, Tuple, Union

import numpy as np

EARTH_RADIUS_KM = 6371.0


def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    radius_km: float = EARTH_RADIUS_KM,
) -> float:
    """Calculate the great-circle distance between two points in kilometers."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    # Numerical stability clamp
    a = min(1.0, max(0.0, a))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return radius_km * c


def haversine_distance_vectorized(
    lat: float,
    lon: float,
    lats: np.ndarray,
    lons: np.ndarray,
    radius_km: float = EARTH_RADIUS_KM,
) -> np.ndarray:
    """Calculate distances from a single point to an array of coordinate pairs in km."""
    phi1 = np.radians(lat)
    phi2 = np.radians(lats)
    delta_phi = np.radians(lats - lat)
    delta_lambda = np.radians(lons - lon)

    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return radius_km * c


def bounding_box(
    lat: float,
    lon: float,
    radius_km: float,
) -> Tuple[float, float, float, float]:
    """Calculate approximate bounding box (min_lat, min_lon, max_lat, max_lon) for a radius."""
    radius_km = max(0.0, float(radius_km))
    lat_delta = radius_km / 111.0
    cos_lat = math.cos(math.radians(lat))
    # Avoid division by zero near poles
    lon_delta = radius_km / (111.0 * max(0.01, abs(cos_lat)))

    min_lat = max(-90.0, lat - lat_delta)
    max_lat = min(90.0, lat + lat_delta)
    min_lon = max(-180.0, lon - lon_delta)
    max_lon = min(180.0, lon + lon_delta)

    return (min_lat, min_lon, max_lat, max_lon)


def point_in_bbox(
    lat: float,
    lon: float,
    bbox: Union[Tuple[float, float, float, float], Sequence[float]],
) -> bool:
    """Check if a coordinate point lies inside a bounding box (min_lat, min_lon, max_lat, max_lon)."""
    min_lat, min_lon, max_lat, max_lon = bbox
    return (min_lat <= lat <= max_lat) and (min_lon <= lon <= max_lon)
