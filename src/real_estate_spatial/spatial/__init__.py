"""Spatial calculation, indexing, and feature engineering utilities."""

from real_estate_spatial.spatial.coordinates import (
    EARTH_RADIUS_KM,
    bounding_box,
    haversine_distance,
    haversine_distance_vectorized,
    point_in_bbox,
)
from real_estate_spatial.spatial.features import SpatialFeatureExtractor
from real_estate_spatial.spatial.index import GridIndex, KDTreeIndex, SpatialIndex

__all__ = [
    "EARTH_RADIUS_KM",
    "bounding_box",
    "haversine_distance",
    "haversine_distance_vectorized",
    "point_in_bbox",
    "SpatialIndex",
    "KDTreeIndex",
    "GridIndex",
    "SpatialFeatureExtractor",
]
