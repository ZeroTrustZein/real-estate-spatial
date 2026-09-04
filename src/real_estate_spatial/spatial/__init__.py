"""Spatial calculation, indexing, and feature engineering utilities."""

from real_estate_spatial.spatial.coordinates import (
    EARTH_RADIUS_KM,
    bounding_box,
    haversine_distance,
    haversine_distance_vectorized,
    point_in_bbox,
)
from real_estate_spatial.spatial.features import SpatialFeatureExtractor
from real_estate_spatial.spatial.index import (
    GridIndex,
    KDTreeIndex,
    SpatialIndex,
    chord_distance_to_great_circle_km,
    great_circle_km_to_chord_distance,
    unpack_bbox,
)
from real_estate_spatial.spatial.persistence import (
    SpatialIndexSerializer,
    SpatialQueryCache,
)

__all__ = [
    "EARTH_RADIUS_KM",
    "bounding_box",
    "haversine_distance",
    "haversine_distance_vectorized",
    "point_in_bbox",
    "SpatialIndex",
    "KDTreeIndex",
    "GridIndex",
    "chord_distance_to_great_circle_km",
    "great_circle_km_to_chord_distance",
    "unpack_bbox",
    "SpatialFeatureExtractor",
    "SpatialIndexSerializer",
    "SpatialQueryCache",
]
