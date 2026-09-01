"""real-estate-spatial: Spatial property valuation and automated comparables engine."""

from real_estate_spatial.comps.adjustments import CompAdjustmentCalculator
from real_estate_spatial.comps.engine import CompsEngine
from real_estate_spatial.comps.scorer import SimilarityScorer
from real_estate_spatial.data.generator import SyntheticRealEstateGenerator
from real_estate_spatial.data.loader import PropertyDataLoader
from real_estate_spatial.models.base import BaseValuationModel, ValuationModel
from real_estate_spatial.models.hedonic import HedonicSpatialRegression
from real_estate_spatial.models.spatial_gbm import SpatialGradientBoostingModel
from real_estate_spatial.spatial.coordinates import (
    EARTH_RADIUS_KM,
    bounding_box,
    haversine_distance,
    haversine_distance_vectorized,
    point_in_bbox,
)
from real_estate_spatial.spatial.features import SpatialFeatureExtractor
from real_estate_spatial.spatial.index import GridIndex, KDTreeIndex, SpatialIndex
from real_estate_spatial.types import (
    CompCandidate,
    Coordinates,
    ModelMetrics,
    Property,
    ValuationResult,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Coordinates",
    "Property",
    "ValuationResult",
    "CompCandidate",
    "ModelMetrics",
    "SpatialIndex",
    "KDTreeIndex",
    "GridIndex",
    "SpatialFeatureExtractor",
    "EARTH_RADIUS_KM",
    "bounding_box",
    "haversine_distance",
    "haversine_distance_vectorized",
    "point_in_bbox",
    "SimilarityScorer",
    "CompAdjustmentCalculator",
    "CompsEngine",
    "BaseValuationModel",
    "ValuationModel",
    "HedonicSpatialRegression",
    "SpatialGradientBoostingModel",
    "SyntheticRealEstateGenerator",
    "PropertyDataLoader",
]
